//===-- CudaKernelNoalias.cpp - CUDA Kernel Noalias Optimization --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass clones CUDA kernel functions to create noalias versions based on
// profile-guided selection of pointer arguments (including struct members).
// It reads per-launch records from profile.kernels[] and computes the best
// subset of pointer parameters that are frequently non-aliasing, then clones
// the kernel and adds noalias attributes to the selected top-level parameters.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/CudaKernelNoalias/CudaKernelNoalias.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Error.h"
#include "llvm/Analysis/CudaKernelAnalysis.h"
#include <algorithm>
#include <string>
#include <map>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

namespace {
struct FileLockGuard {
  int Fd; bool Ok;
  FileLockGuard(const std::string &Path) : Fd(-1), Ok(false) {
    std::string LockPath = Path + ".lock";
    Fd = ::open(LockPath.c_str(), O_CREAT | O_RDWR, 0666);
    if (Fd != -1) {
      int Ret;
      do { Ret = ::flock(Fd, LOCK_EX); } while (Ret != 0 && errno == EINTR);
      if (Ret == 0) Ok = true; else { ::close(Fd); Fd = -1; }
    }
  }
  ~FileLockGuard() { if (Fd != -1) { ::flock(Fd, LOCK_UN); ::close(Fd); } }
  bool acquired() const { return Ok; }
};

struct FileSharedLockGuard {
  int Fd; bool Ok;
  FileSharedLockGuard(const std::string &Path) : Fd(-1), Ok(false) {
    std::string LockPath = Path + ".lock";
    Fd = ::open(LockPath.c_str(), O_CREAT | O_RDWR, 0666);
    if (Fd != -1) {
      int Ret;
      do { Ret = ::flock(Fd, LOCK_SH); } while (Ret != 0 && errno == EINTR);
      if (Ret == 0) Ok = true; else { ::close(Fd); Fd = -1; }
    }
  }
  ~FileSharedLockGuard() { if (Fd != -1) { ::flock(Fd, LOCK_UN); ::close(Fd); } }
  bool acquired() const { return Ok; }
};
}

using namespace llvm;

#define DEBUG_TYPE "cuda-kernel-noalias"

// Profile path (reuse existing option name for compatibility)
static cl::opt<std::string> CudaProfilePath(
    "cuda-kernel-profile",
    cl::desc("Path to the CUDA kernel profile log file"),
    cl::value_desc("filename"), cl::init(""));

static cl::opt<bool> CudaNoaliasDebug(
    "cuda-kernel-noalias-debug",
    cl::desc("Enable debug prints for CudaKernelNoalias pass"), cl::init(false));

static cl::opt<std::string> CudaNoaliasSelectedOut(
    "cuda-kernel-noalias-selected-out",
    cl::desc("Write selected noalias parameters to JSON file for host-side"),
    cl::value_desc("filename"), cl::init(""));

// Return true if the given function is a PTX kernel entry
static bool isKernelFunction(const Function &F) {
  return F.getCallingConv() == CallingConv::PTX_Kernel;
}

#if 0
// Kernel launches profile for noalias (flattened pointer-path -> alias flag)
struct KernelNoaliasProfile {
  std::string name;
  std::map<unsigned, std::string> argIndexToName; // optional for pretty output
  std::vector<StringMap<int>> launches; // path -> alias (0 means no alias)
};

// Selected item with full indices path
struct SelectedNoaliasItem {
  SmallVector<unsigned, 4> Indices; // full path, e.g. [arg, member, ...]
  double Ratio = 0.0;
};

static std::map<std::string, KernelNoaliasProfile> KernelProfiles;
static std::map<std::string, std::vector<SelectedNoaliasItem>> SelectedByKernel;
#endif

// Helper: recursively collect pointer member paths and their alias flag into OutMap
static void collectPointerAliases(const json::Object &Obj, unsigned ArgIndex,
                                  StringMap<int> &OutMap) {
  // This function is called for a top-level parameter from the profile.
  bool isStruct = false;
  if (auto ValueNode = Obj.get("value")) {
    if (auto *Arr = ValueNode->getAsArray()) {
      if (!Arr->empty() && (*Arr)[0].getAsObject()) {
        isStruct = true;
      }
    }
  }

  if (!isStruct) {
    // It's a top-level pointer, not a struct.
    if (auto Ty = Obj.getString("type")) {
      if (Ty->contains('*')) {
        int AliasFlag = -1;
        if (auto A = Obj.get("alias")) {
          if (auto AI = A->getAsInteger()) AliasFlag = (int)*AI;
        }
        std::string Key = std::to_string(ArgIndex);
        OutMap[Key] = AliasFlag;
        OutMap[Key + ".0"] = AliasFlag;
      }
    }
    return;
  }

  // It's a struct. Use recursive logic to find member offsets.
  std::function<void(const json::Object &, uint64_t)> recurse;
  recurse = [&](const json::Object &CurrentObj, uint64_t BaseOffset) {
    uint64_t CurrentOffset = BaseOffset;
    if (auto RelOffset = CurrentObj.getInteger("offset")) {
      CurrentOffset += *RelOffset;
    }

    // If it's a pointer type, record its alias info at the current offset.
    if (auto Ty = CurrentObj.getString("type")) {
      if (Ty->contains('*')) {
        int AliasFlag = -1; // Default to unknown
        if (auto A = CurrentObj.get("alias")) {
          if (auto AI = A->getAsInteger()) AliasFlag = (int)*AI;
        }
        std::string Path = std::to_string(ArgIndex) + "." + std::to_string(CurrentOffset);
        OutMap[Path] = AliasFlag;
      }
    }

    // If the value is a struct (array of members), recurse.
    if (auto ValueNode = CurrentObj.get("value")) {
      if (auto MembersArr = ValueNode->getAsArray()) {
        for (const json::Value &Elem : *MembersArr) {
          if (auto *Child = Elem.getAsObject()) {
            recurse(*Child, CurrentOffset);
          }
        }
      }
    }
  };
  recurse(Obj, 0);
}

// Parse JSON profile launches (kernels[]) and build KernelProfiles
bool CudaKernelNoaliasPass::parseProfileLog() {
  KernelProfiles.clear();

  auto getEffectiveProfilePath = []() -> std::string {
    if (!CudaProfilePath.empty()) return CudaProfilePath;
    if (const char *Env = std::getenv("CUDA_KERNEL_PROFILE")) return std::string(Env);
    return std::string();
  };

  std::string EffectivePath = getEffectiveProfilePath();
  if (EffectivePath.empty()) {
    errs() << "Warning: No profile log file specified. Use -mllvm -cuda-kernel-profile=<file> or set CUDA_KERNEL_PROFILE\n";
    return false;
  }

  FileSharedLockGuard ReadLock(EffectivePath);

  auto BufferOrErr = MemoryBuffer::getFile(EffectivePath);
  if (!BufferOrErr) {
    errs() << "Error: Cannot open profile log file: " << EffectivePath << "\n";
    return false;
  }

  StringRef Content = BufferOrErr.get()->getBuffer();
  auto Parsed = json::parse(Content);
  if (!Parsed) {
    errs() << "Error: Failed to parse JSON profile log: " << toString(Parsed.takeError()) << "\n";
    return false;
  }

  json::Object *RootObj = Parsed->getAsObject();
  if (!RootObj) {
    errs() << "Error: JSON root is not an object\n";
    return false;
  }

  if (auto KernelsArr = RootObj->getArray("kernels")) {
    for (json::Value &KVal : *KernelsArr) {
      json::Object *KObj = KVal.getAsObject();
      if (!KObj) continue;
      auto NameVal = KObj->getString("device_side_name");
      if (!NameVal)
        NameVal = KObj->getString("name");
      if (!NameVal) continue;
      KernelNoaliasProfile &KP = KernelProfiles[NameVal->str()];
      KP.name = NameVal->str();

      // record arg index to name (optional)
      if (auto ParamsArr = KObj->getArray("params")) {
        for (json::Value &PVal : *ParamsArr) {
          if (json::Object *PObj = PVal.getAsObject()) {
            if (auto IndexVal = PObj->get("index")) {
              if (auto IndexInt = IndexVal->getAsInteger()) {
                if (auto N = PObj->getString("name"))
                  KP.argIndexToName[(unsigned)*IndexInt] = N->str();
              }
            }
          }
        }
      }

      // collect pointer alias flags per launch (flattened by path)
      StringMap<int> LaunchMap;
      if (auto ParamsArr = KObj->getArray("params")) {
        for (json::Value &PVal : *ParamsArr) {
          if (auto *PObj = PVal.getAsObject()) {
            if (auto Index = PObj->getInteger("index")) {
              collectPointerAliases(*PObj, *Index, LaunchMap);
            }
          }
        }
      }
      if (!LaunchMap.empty()) KP.launches.push_back(std::move(LaunchMap));
    }
  }

  if (CudaNoaliasDebug) {
    dbgs() << "[CudaKernelNoalias] profile loaded from '" << EffectivePath
           << "' kernels_parsed=" << KernelProfiles.size()
           << "\n";
    for (const auto &KV : KernelProfiles) {
      dbgs() << "  - name='" << KV.first << "' launches=" << KV.second.launches.size() << "\n";
    }
  }

  return true;
}

// Build a string path from indices vector
static std::string buildPath(const SmallVector<unsigned,4> &Idx) {
  std::string P = std::to_string((int)Idx[0]);
  for (size_t i = 1; i < Idx.size(); ++i) P += "." + std::to_string((int)Idx[i]);
  return P;
}

// Compute best subset per kernel using analysis weights and launches alias flags
void CudaKernelNoaliasPass::computeBestSelections(Module &M, ModuleAnalysisManager &AM) {
  SelectedByKernel.clear();

  // Access FunctionAnalysisManager from ModuleAnalysisManager
  auto &FAMProxy = AM.getResult<FunctionAnalysisManagerModuleProxy>(M);
  FunctionAnalysisManager &FAM = FAMProxy.getManager();

  for (Function &F : M) {
    if (!isKernelFunction(F)) continue;

    auto KIt = KernelProfiles.find(F.getName().str());
    if (KIt == KernelProfiles.end()) {
      if (CudaNoaliasDebug)
        dbgs() << "[CudaKernelNoalias] skip kernel '" << F.getName()
               << "': no profile entry\n";
      continue;
    }
    const KernelNoaliasProfile &KP = KIt->second;
    if (KP.launches.empty()) {
      if (CudaNoaliasDebug)
        dbgs() << "[CudaKernelNoalias] skip kernel '" << F.getName()
               << "': profile launches empty\n";
      continue;
    }

    // Candidates from analysis (pointer weights > 0), include struct members
    auto AR = FAM.getResult<CudaKernelAnalysis>(F);
    struct Candidate { SmallVector<unsigned,4> Indices; std::string Path; double Weight; };
    std::vector<Candidate> Cands;
    unsigned Eligible = 0;
    for (const auto &Entry : AR.PointerWeights) {
      const ParameterInfo &PI = Entry.first;
      int W = Entry.second;
      if (W <= 0) continue; else ++Eligible;
      Candidate C;
      C.Indices = PI.Indices;
      C.Path = buildPath(C.Indices);
      C.Weight = (double)W;
      Cands.push_back(std::move(C));
    }
    if (CudaNoaliasDebug) {
      dbgs() << "[CudaKernelNoalias] kernel='" << F.getName()
             << "' analysis_candidates(pointer_weights>0)=" << Cands.size()
             << " (total_pointer_entries=" << Eligible << ")\n";
    }
    if (Cands.empty()) continue;

    // Enumerate non-empty subsets
    const unsigned Num = Cands.size();
    double BestScore = -1.0;
    unsigned BestMask = 0;
    SmallVector<SmallVector<unsigned,4>, 8> BestIdxPaths;
    SmallVector<double, 8> BestSingleRatio;

    for (unsigned Mask = 1; Mask < (1u << Num); ++Mask) {
      SmallVector<std::string, 8> SubPaths;
      SmallVector<double, 8> SubWeights;
      SmallVector<SmallVector<unsigned,4>, 8> SubIdx;
      for (unsigned i = 0; i < Num; ++i) if (Mask & (1u << i)) {
        SubPaths.push_back(Cands[i].Path);
        SubWeights.push_back(Cands[i].Weight);
        SubIdx.push_back(Cands[i].Indices);
      }

      // Ratio: launch count where all selected paths exist and alias==0
      size_t Sat = 0;
      for (const auto &L : KP.launches) {
        bool Ok = true;
        for (const auto &P : SubPaths) {
          auto It = L.find(P);
          if (It == L.end()) { Ok = false; break; }
          int AliasFlag = It->second;
          if (AliasFlag != 0) { Ok = false; break; }
        }
        if (Ok) ++Sat;
      }
      double Ratio = KP.launches.empty() ? 0.0 : (double)Sat / (double)KP.launches.size();
      double Score = 0.0; for (double W : SubWeights) Score += W * Ratio;

      auto PrintSubset = [&](raw_ostream &OS) {
        OS << "[CudaKernelNoalias] kernel=" << F.getName()
           << " subset_mask=0x" << Twine::utohexstr(Mask)
           << " ratio=" << Ratio << " score=" << Score
           << " sat=" << Sat << "/" << KP.launches.size() << " paths={";
        for (size_t i = 0; i < SubPaths.size(); ++i) {
          if (i) OS << ", "; OS << SubPaths[i];
        }
        OS << "}\n";
      };
      if (CudaNoaliasDebug) { PrintSubset(dbgs()); } else { LLVM_DEBUG(PrintSubset(dbgs())); }

      if (Score > BestScore || (Score == BestScore && BestMask != 0 &&
                                __builtin_popcount(BestMask) < __builtin_popcount(Mask))) {
        BestScore = Score; BestMask = Mask; BestIdxPaths = std::move(SubIdx);
        // Compute per-item single ratio for reporting
        BestSingleRatio.clear(); BestSingleRatio.reserve(SubPaths.size());
        for (const auto &P : SubPaths) {
          size_t Cnt = 0; for (const auto &L : KP.launches) {
            auto It = L.find(P); if (It != L.end() && It->second == 0) ++Cnt;
          }
          BestSingleRatio.push_back(KP.launches.empty() ? 0.0 : (double)Cnt / (double)KP.launches.size());
        }
      }
    }

    if (BestMask == 0 || BestIdxPaths.empty()) continue;

    // Emit selection result
    std::vector<SelectedNoaliasItem> Items;
    for (size_t i = 0; i < BestIdxPaths.size(); ++i) {
      SelectedNoaliasItem It; It.Indices = BestIdxPaths[i]; It.Ratio = BestSingleRatio[i];
      Items.push_back(std::move(It));
    }

    auto PrintBest = [&](raw_ostream &OS) {
      OS << "[CudaKernelNoalias] kernel=" << F.getName()
         << " best_mask=0x" << Twine::utohexstr(BestMask)
         << " best_score=" << BestScore << " selected={";
      for (size_t i = 0; i < Items.size(); ++i) {
        if (i) OS << ", ";
        OS << "path=" << buildPath(Items[i].Indices) << "(ratio=" << Items[i].Ratio << ")";
      }
      OS << "}\n";
    };
    if (CudaNoaliasDebug) { PrintBest(dbgs()); } else { LLVM_DEBUG(PrintBest(dbgs())); }

    SelectedByKernel[F.getName().str()] = std::move(Items);
  }
}

// Dump selections to JSON file (array form), or merge into profile if no out given
static void dumpSelectionsToJson(const std::map<std::string, std::vector<SelectedNoaliasItem>> &Selected, 
                                 const std::string &OutPath) {
  if (OutPath.empty() || Selected.empty()) return;

  FileLockGuard Lock(OutPath);
  if (!Lock.acquired()) { errs() << "Warning: cannot acquire lock for '" << OutPath << "', skipping write.\n"; return; }

  // Load existing selections to merge
  std::map<std::string, std::vector<SelectedNoaliasItem>> Existing;
  if (auto BufOrErr = MemoryBuffer::getFile(OutPath)) {
    if (auto Parsed = json::parse(BufOrErr.get()->getBuffer())) {
      if (auto *RootObj = Parsed->getAsObject()) {
        if (auto *SelObj = RootObj->getObject("cuda_noalias_selected")) {
          for (auto &KV : *SelObj) {
            std::string KName = KV.first.str();
            if (auto *Arr = KV.second.getAsArray()) {
              for (const auto &V : *Arr) if (auto *O = V.getAsObject()) {
                SelectedNoaliasItem It; if (auto *Idxs = O->getArray("indices")) {
                  for (const auto &X : *Idxs) if (auto I = X.getAsInteger()) It.Indices.push_back((unsigned)*I);
                }
                if (auto R = O->getNumber("ratio")) It.Ratio = *R; if (!It.Indices.empty()) Existing[KName].push_back(std::move(It));
              }
            }
          }
        } else if (auto *SelArr = RootObj->getArray("cuda_noalias_selected")) {
          for (const auto &Elem : *SelArr) if (auto *Obj = Elem.getAsObject()) {
            std::string KName; if (auto KS = Obj->getString("name")) KName = KS->str(); else if (auto KS2 = Obj->getString("kernel")) KName = KS2->str();
            if (KName.empty()) continue; if (auto *Arr = Obj->getArray("selected")) {
              for (const auto &V : *Arr) if (auto *O = V.getAsObject()) {
                SelectedNoaliasItem It; if (auto *Idxs = O->getArray("indices")) {
                  for (const auto &X : *Idxs) if (auto I = X.getAsInteger()) It.Indices.push_back((unsigned)*I);
                }
                if (auto R = O->getNumber("ratio")) It.Ratio = *R; if (!It.Indices.empty()) Existing[KName].push_back(std::move(It));
              }
            }
          }
        }
      }
    }
  }

  auto indicesEqual = [](const SmallVector<unsigned,4> &A, const SmallVector<unsigned,4> &B) -> bool {
    if (A.size() != B.size()) return false;
    for (size_t i = 0; i < A.size(); ++i) if (A[i] != B[i]) return false;
    return true;
  };

  // Merge
  for (const auto &KV : Selected) {
    auto &Dst = Existing[KV.first];
    for (const auto &It : KV.second) {
      bool Replaced = false; for (auto &Old : Dst) { if (indicesEqual(Old.Indices, It.Indices)) { Old.Ratio = It.Ratio; Replaced = true; break; } } 
      if (!Replaced) Dst.push_back(It);
    }
  }

  // Serialize array form
  json::Array Kernels;
  for (const auto &KV : Existing) {
    json::Object KObj; KObj["name"] = KV.first; json::Array Items;
    for (const auto &It : KV.second) { json::Object O; json::Array Idxs; for (unsigned idx : It.Indices) Idxs.push_back((int64_t)idx);
      O["indices"] = std::move(Idxs); O["value"] = ""; O["ratio"] = It.Ratio; Items.push_back(std::move(O)); }
    KObj["selected"] = std::move(Items); Kernels.push_back(std::move(KObj));
  }
  json::Object Root; Root["cuda_noalias_selected"] = std::move(Kernels);
  std::error_code EC; raw_fd_ostream OS(OutPath, EC); if (EC) { errs() << "Failed to write cuda-kernel-noalias-selected-out to '" << OutPath << "': " << EC.message() << "\n"; return; }
  OS << formatv("{0:2}\n", json::Value(std::move(Root)));
}

static void mergeSelectionsIntoProfile(const std::map<std::string, std::vector<SelectedNoaliasItem>> &Selected,
                                       const std::string &ProfilePath) {
  if (ProfilePath.empty() || Selected.empty()) return;

  FileLockGuard Lock(ProfilePath);
  if (!Lock.acquired()) { errs() << "Warning: cannot acquire lock for '" << ProfilePath << "', skipping write.\n"; return; }

  // Load existing
  std::map<std::string, std::vector<SelectedNoaliasItem>> Existing;
  json::Object Root;
  if (auto BufOrErr = MemoryBuffer::getFile(ProfilePath)) {
    if (auto Parsed = json::parse(BufOrErr.get()->getBuffer())) {
      if (auto *Obj = Parsed->getAsObject()) {
        Root = *Obj;
        if (auto *SelObj = Root.getObject("cuda_noalias_selected")) {
          for (auto &KV : *SelObj) {
            std::string KName = KV.first.str();
            if (auto *Arr = KV.second.getAsArray()) {
              for (const auto &V : *Arr) if (auto *O = V.getAsObject()) {
                SelectedNoaliasItem It; if (auto *Idxs = O->getArray("indices")) {
                  for (const auto &X : *Idxs) if (auto I = X.getAsInteger()) It.Indices.push_back((unsigned)*I);
                }
                if (auto R = O->getNumber("ratio")) It.Ratio = *R; if (!It.Indices.empty()) Existing[KName].push_back(std::move(It));
              }
            }
          }
        } else if (auto *SelArr = Root.getArray("cuda_noalias_selected")) {
          for (const auto &Elem : *SelArr) if (auto *Obj2 = Elem.getAsObject()) {
            std::string KName; if (auto KS = Obj2->getString("name")) KName = KS->str(); else if (auto KS2 = Obj2->getString("kernel")) KName = KS2->str();
            if (KName.empty()) continue; if (auto *Arr = Obj2->getArray("selected")) {
              for (const auto &V : *Arr) if (auto *O = V.getAsObject()) {
                SelectedNoaliasItem It; if (auto *Idxs = O->getArray("indices")) {
                  for (const auto &X : *Idxs) if (auto I = X.getAsInteger()) It.Indices.push_back((unsigned)*I);
                }
                if (auto R = O->getNumber("ratio")) It.Ratio = *R; if (!It.Indices.empty()) Existing[KName].push_back(std::move(It));
              }
            }
          }
        }
      }
    }
  }

  auto indicesEqual = [](const SmallVector<unsigned,4> &A, const SmallVector<unsigned,4> &B) -> bool {
    if (A.size() != B.size()) return false; for (size_t i = 0; i < A.size(); ++i) if (A[i] != B[i]) return false; return true;
  };

  // Merge incoming
  for (const auto &KV : Selected) {
    auto &Dst = Existing[KV.first];
    for (const auto &It : KV.second) {
      bool Replaced = false; for (auto &Old : Dst) { if (indicesEqual(Old.Indices, It.Indices)) { Old.Ratio = It.Ratio; Replaced = true; break; } } 
      if (!Replaced) Dst.push_back(It);
    }
  }

  // Serialize back as array form
  json::Array Kernels;
  for (const auto &KV : Existing) {
    json::Object KObj; KObj["name"] = KV.first; json::Array Items;
    for (const auto &It : KV.second) {
      json::Object O; json::Array Idxs; for (unsigned idx : It.Indices) Idxs.push_back((int64_t)idx);
      O["indices"] = std::move(Idxs); O["value"] = ""; O["ratio"] = It.Ratio; Items.push_back(std::move(O));
    }
    KObj["selected"] = std::move(Items); Kernels.push_back(std::move(KObj));
  }
  Root["cuda_noalias_selected"] = std::move(Kernels);
  std::error_code EC; raw_fd_ostream OS(ProfilePath, EC);
  if (EC) { errs() << "Failed to update profile '" << ProfilePath << "' with noalias selections: " << EC.message() << "\n"; return; }
  OS << formatv("{0:2}\n", json::Value(std::move(Root)));
}

// Helper to find all memory accesses (loads/stores) from a given pointer value.
// This function traverses through GEPs and bitcasts.
static void findMemoryAccesses(Value *Ptr, SmallVectorImpl<Instruction *> &Accesses) {
    SmallVector<Value *, 16> Worklist;
    Worklist.push_back(Ptr);
    SmallPtrSet<Value *, 16> Visited;

    while (!Worklist.empty()) {
        Value *V = Worklist.pop_back_val();
        if (!Visited.insert(V).second) continue;

        for (User *U : V->users()) {
            if (auto *I = dyn_cast<Instruction>(U)) {
                if (isa<LoadInst>(I) || isa<StoreInst>(I)) {
                    Accesses.push_back(I);
                } else if (isa<GetElementPtrInst>(I) || isa<BitCastInst>(I)) {
                    Worklist.push_back(I);
                } else if (isa<CallInst>(I) || isa<CallBrInst>(I)) {
                    // TODO: Also consider calls/invokes as potential memory accesses if they use the pointer. Such as
                    // inline ptx for memory access, now just add the call inst into Accesses.
                    Accesses.push_back(I);
                }
            }
        }
    }
}

// Apply noalias metadata to selected pointers (top-level or nested)
// TODO: this func now only performe ptr(from cuda kernel analysis pass) noalias with all the other ptrs from cuda 
// kernel analysis pass, but idealy it should consider all ptrs and their memory accesses in the kernel. 
static void performNoaliasTransformation(Function &F,
                                         const std::vector<SelectedNoaliasItem> &Items) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  LLVMContext &Ctx = F.getContext();

  // Collect all nested pointer values that need to be wrapped with metadata.
  SmallVector<Value*, 16> PointersToWrap;
  for (const auto &Item : Items) {
    // Only process struct members (path size >= 2). Top-level pointers are
    // handled with parameter attributes.
    if (Item.Indices.size() < 2) continue;

    unsigned ArgIndex = Item.Indices[0];
    if (ArgIndex >= F.arg_size()) continue;
    Argument *Arg = F.getArg(ArgIndex);
    uint64_t Offset = Item.Indices[1];

    // This logic handles the first member (offset 0) and subsequent members.
    bool FoundAtOffset0 = false;
    if (Offset == 0) {
        for (User *U : Arg->users()) {
            if (auto *LI = dyn_cast<LoadInst>(U)) {
                if (LI->getType()->isPointerTy()) {
                    PointersToWrap.push_back(LI);
                    FoundAtOffset0 = true;
                }
            }
        }
    }
    if (FoundAtOffset0) continue;

    for (User *U : Arg->users()) {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        APInt GEP_Offset(DL.getPointerSizeInBits(GEP->getPointerAddressSpace()), 0);
        if (GEP->accumulateConstantOffset(DL, GEP_Offset)) {
          if (GEP_Offset.getZExtValue() == Offset) {
            SmallVector<Value *, 4> Worklist;
            Worklist.push_back(GEP);
            SmallPtrSet<Value *, 4> Visited;
            while (!Worklist.empty()) {
              Value *V = Worklist.pop_back_val();
              if (!Visited.insert(V).second) continue;
              for (User *GEPUser : V->users()) {
                if (isa<BitCastInst>(GEPUser)) {
                  Worklist.push_back(GEPUser);
                } else if (auto *LI = dyn_cast<LoadInst>(GEPUser)) {
                  if (LI->getType()->isPointerTy()) PointersToWrap.push_back(LI);
                }
              }
            }
          }
        }
      }
    }
  }

  if (PointersToWrap.size() < 2) return;

  MDBuilder MDB(Ctx);
  MDNode *Domain = MDB.createAnonymousAliasScopeDomain("cuda.noalias.domain");

  // Create a stable mapping from each pointer to a unique scope.
  DenseMap<Value*, Metadata*> PtrToScopeMap;
  for (Value *PtrVal : PointersToWrap) {
      std::string Name = "scope.";
      if (PtrVal->hasName()) {
          Name += PtrVal->getName().str();
      } else {
          Name += std::to_string(reinterpret_cast<uintptr_t>(PtrVal));
      }
      PtrToScopeMap[PtrVal] = MDB.createAnonymousAliasScope(Domain, Name);
  }

  // For each pointer, find its memory accesses and tag them.
  for (Value *PtrVal : PointersToWrap) {
    SmallVector<Instruction*, 16> Accesses;
    findMemoryAccesses(PtrVal, Accesses);

    Metadata *ThisScope = PtrToScopeMap[PtrVal];

    SmallVector<Metadata*, 16> NoAliasList;
    for (auto const& [OtherPtr, OtherScope] : PtrToScopeMap) {
        if (PtrVal == OtherPtr) continue;
        NoAliasList.push_back(OtherScope);
    }
    if (NoAliasList.empty()) continue;

    MDNode* ScopeMD = MDNode::get(Ctx, {ThisScope});
    MDNode* NoAliasMD = MDNode::get(Ctx, NoAliasList);

    for (Instruction* Inst : Accesses) {
      // FIXME: If the instruction is a call inst such as inline ptx, is there way to add the scope metadata to the 
      // real memory access? for now just add the scope metadata to the call inst.
        Inst->setMetadata(LLVMContext::MD_alias_scope, ScopeMD);
        Inst->setMetadata(LLVMContext::MD_noalias, NoAliasMD);
    }
  }
}

// Clone kernel and apply noalias transformation
static Function *cloneAndApplyNoalias(Function &OrigF,
                                        const std::vector<SelectedNoaliasItem> &Items) {
  ValueToValueMapTy VMap; Function *ClonedF = CloneFunction(&OrigF, VMap);
  std::string NewName = OrigF.getName().str() + "_noalias"; ClonedF->setName(NewName);

  // Apply noalias transformation using metadata for nested pointers.
  performNoaliasTransformation(*ClonedF, Items);

  // Apply noalias attribute to selected top-level pointer arguments.
  for (const auto &Item : Items) {
      if (Item.Indices.size() == 1) {
          unsigned ArgIndex = Item.Indices[0];
          if (ArgIndex < ClonedF->arg_size() && ClonedF->getArg(ArgIndex)->getType()->isPointerTy()) {
              ClonedF->addParamAttr(ArgIndex, Attribute::NoAlias);
          }
      }
  }

  // Copy nvvm.annotations kernel tag
  Module *M = OrigF.getParent(); NamedMDNode *NMD = M->getOrInsertNamedMetadata("nvvm.annotations");
  for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i) {
    const MDNode *MD = NMD->getOperand(i);
    if (MD->getNumOperands() >= 3) {
      if (auto *FMD = mdconst::dyn_extract_or_null<Function>(MD->getOperand(0))) {
        if (FMD == &OrigF) {
          if (auto *Str = dyn_cast<MDString>(MD->getOperand(1))) {
            if (Str->getString() == "kernel") {
              LLVMContext &Ctx = M->getContext();
              Metadata *MDVals[] = { ValueAsMetadata::get(ClonedF), MDString::get(Ctx, "kernel"), MD->getOperand(2) };
              MDNode *NewMD = MDNode::get(Ctx, MDVals); NMD->addOperand(NewMD); break;
            }
          }
        }
      }
    }
  }
  return ClonedF;
}

PreservedAnalyses CudaKernelNoaliasPass::run(Module &M, ModuleAnalysisManager &AM) {
  if (!parseProfileLog()) {
    errs() << "Warning: Failed to parse profile log, skipping noalias optimization\n";
    return PreservedAnalyses::all();
  }

  // Compute selections
  computeBestSelections(M, AM);

  bool Changed = false; std::vector<Function*> Kernels;
  for (Function &F : M) if (isKernelFunction(F)) Kernels.push_back(&F);

  for (Function *F : Kernels) {
    auto It = SelectedByKernel.find(F->getName().str());
    if (It == SelectedByKernel.end() || It->second.empty()) continue;

    Function *Clone = cloneAndApplyNoalias(*F, It->second);
    (void)Clone; Changed = true;
    LLVM_DEBUG(dbgs() << "Cloned kernel " << F->getName() << " to " << Clone->getName()
                      << " with noalias on " << It->second.size() << " selected entries\n");
  }

  if (!CudaNoaliasSelectedOut.empty()) {
    dumpSelectionsToJson(SelectedByKernel, CudaNoaliasSelectedOut);
  } else {
    std::string EffectivePath;
    if (!CudaProfilePath.empty()) EffectivePath = CudaProfilePath;
    else if (const char *Env = std::getenv("CUDA_KERNEL_PROFILE")) EffectivePath = Env;
    if (!EffectivePath.empty()) mergeSelectionsIntoProfile(SelectedByKernel, EffectivePath);
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
