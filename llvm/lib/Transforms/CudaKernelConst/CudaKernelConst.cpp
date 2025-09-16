//===-- CudaKernelConst.cpp - CUDA Kernel Const Propagation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass clones CUDA kernel functions to create constant propagation
// versions based on profiling information for common scalar and dimension
// values.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/CudaKernelConst/CudaKernelConst.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Error.h"
#include "llvm/Analysis/CudaKernelAnalysis.h"
#include <regex>
#include <cstdlib>
#include <cmath>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

namespace {
struct FileLockGuard {
  int Fd;
  bool Ok;
  FileLockGuard(const std::string &Path) : Fd(-1), Ok(false) {
    std::string LockPath = Path + ".lock";
    Fd = ::open(LockPath.c_str(), O_CREAT | O_RDWR, 0666);
    if (Fd != -1) {
      int Ret;
      do { Ret = ::flock(Fd, LOCK_EX); } while (Ret != 0 && errno == EINTR);
      if (Ret == 0) Ok = true; else { ::close(Fd); Fd = -1; }
    }
  }
  ~FileLockGuard() {
    if (Fd != -1) { ::flock(Fd, LOCK_UN); ::close(Fd); }
  }
  bool acquired() const { return Ok; }
};

struct FileSharedLockGuard {
  int Fd;
  bool Ok;
  FileSharedLockGuard(const std::string &Path) : Fd(-1), Ok(false) {
    std::string LockPath = Path + ".lock";
    Fd = ::open(LockPath.c_str(), O_CREAT | O_RDWR, 0666);
    if (Fd != -1) {
      int Ret;
      do { Ret = ::flock(Fd, LOCK_SH); } while (Ret != 0 && errno == EINTR);
      if (Ret == 0) Ok = true; else { ::close(Fd); Fd = -1; }
    }
  }
  ~FileSharedLockGuard() {
    if (Fd != -1) { ::flock(Fd, LOCK_UN); ::close(Fd); }
  }
  bool acquired() const { return Ok; }
};
} // end anonymous namespace

// Special indices for implicit CUDA parameters
// These are chosen to be large values to avoid collision with real argument indices.
// The values must be kept in sync with CudaKernelAnalysis.cpp
namespace {
static constexpr unsigned SPECIAL_INDEX_BASE = 0xE0000000;
static constexpr unsigned GRID_DIM_X_INDEX  = SPECIAL_INDEX_BASE;
static constexpr unsigned GRID_DIM_Y_INDEX  = SPECIAL_INDEX_BASE - 1;
static constexpr unsigned GRID_DIM_Z_INDEX  = SPECIAL_INDEX_BASE - 2;
static constexpr unsigned BLOCK_DIM_X_INDEX = SPECIAL_INDEX_BASE - 3;
static constexpr unsigned BLOCK_DIM_Y_INDEX = SPECIAL_INDEX_BASE - 4;
static constexpr unsigned BLOCK_DIM_Z_INDEX = SPECIAL_INDEX_BASE - 5;
}

using namespace llvm;

#define DEBUG_TYPE "cuda-kernel-const"

static cl::opt<bool> CudaConstDebug("cuda-kernel-const-debug",
    cl::desc("Enable debug prints for CudaKernelConst pass"), cl::init(false));

// Forward declarations for type inference helpers
static std::string getScalarTypeString(Type *Ty);
static Type *findLoadTypeForArgOffset(const DataLayout &DL, Argument *Arg, uint64_t Offset);
static Type *inferSelectedParamType(Function &F, ArrayRef<unsigned> Indices);

// Optional: write selected parameters to a JSON file for host-side consumption
static cl::opt<std::string> CudaConstSelectedOut(
    "cuda-kernel-const-selected-out",
    cl::desc("Write selected constant parameters to JSON file for host-side"),
    cl::value_desc("filename"), cl::init(""));

// fwd decl
static void dumpSelectionsToJson(const std::map<std::string, std::vector<SelectedParamRecord>> &Selected,
                                 const std::string &OutPath);
static void mergeSelectionsIntoProfile(const std::map<std::string, std::vector<SelectedParamRecord>> &Selected,
                                       const std::string &ProfilePath);

// Helper: whether a scalar value string from profile is supported (numeric)
static bool isSupportedScalarValue(llvm::StringRef S) {
  if (S == "unsupported_scalar_type") return false;
  // integer decimal
  long long iv = 0;
  if (!S.getAsInteger(10, iv)) return true;
  // floating
  char *end = nullptr;
  double dv = std::strtod(S.str().c_str(), &end);
  if (end && *end == '\0') (void)dv; else return false;
  return true;
}

// Command-line option for specifying the profile JSON file path
static cl::opt<std::string> CudaConstProfilePath(
    "cuda-kernel-const-profile",
    cl::desc("Path to the CUDA kernel profile log file for constant propagation"),
    cl::value_desc("filename"),
    cl::init(""));

// Return true if the given function is a PTX kernel entry
static bool isKernelFunction(const Function &F) {
  return F.getCallingConv() == CallingConv::PTX_Kernel;
}

static void collectParamValues(const json::Object &Obj, unsigned ArgIndex,
                               std::map<std::string, std::string> &OutMap) {
  // This function is called for a top-level parameter from the profile.
  // Check if it's a struct by seeing if its "value" is an array of objects.
  bool isStruct = false;
  if (auto ValueNode = Obj.get("value")) {
    if (auto *Arr = ValueNode->getAsArray()) {
      if (!Arr->empty() && (*Arr)[0].getAsObject()) {
        isStruct = true;
      }
    }
  }

  if (!isStruct) {
    // It's a top-level scalar/pointer, not a struct.
    // The path is just its argument index.
    std::string Path = std::to_string(ArgIndex);
    if (auto ValueNode = Obj.get("value")) {
      auto emitKV = [&](const std::string &K, const std::string &V) {
        OutMap[K] = V;
      };
      if (auto VS = ValueNode->getAsString()) {
        if (isSupportedScalarValue(*VS)) {
          emitKV(Path, VS->str());
          emitKV(Path + ".0", VS->str());
        }
      } else if (auto VNum = ValueNode->getAsNumber()) {
        std::string S = std::to_string(*VNum);
        emitKV(Path, S);
        emitKV(Path + ".0", S);
      } else if (auto VInt = ValueNode->getAsInteger()) {
        std::string S = std::to_string(*VInt);
        emitKV(Path, S);
        emitKV(Path + ".0", S);
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

    if (auto ValueNode = CurrentObj.get("value")) {
      if (auto MembersArr = ValueNode->getAsArray()) {
        // It's a struct, recurse on members
        for (const json::Value &Elem : *MembersArr) {
          if (auto *Child = Elem.getAsObject()) {
            recurse(*Child, CurrentOffset);
          }
        }
      } else { // It's a scalar leaf node inside a struct.
        std::string Path = std::to_string(ArgIndex) + "." + std::to_string(CurrentOffset);
        if (auto VS = ValueNode->getAsString()) {
          if (isSupportedScalarValue(*VS)) {
            OutMap[Path] = VS->str();
          }
        } else if (auto VNum = ValueNode->getAsNumber()) {
          OutMap[Path] = std::to_string(*VNum);
        } else if (auto VInt = ValueNode->getAsInteger()) {
          OutMap[Path] = std::to_string(*VInt);
        }
      }
    }
  };

  // Initial call on the top-level struct object itself.
  recurse(Obj, 0);
}

// Parse the JSON profile log and populate KernelProfiles
bool CudaKernelConstPass::parseProfileLog() {
  std::string ProfilePath = CudaConstProfilePath;
  if (ProfilePath.empty()) {
    if (const char *Env = std::getenv("CUDA_KERNEL_PROFILE")) {
      ProfilePath = Env;
    }
  }

  if (ProfilePath.empty()) {
    errs() << "Warning: No profile log file specified for constant propagation. Use -mllvm -cuda-kernel-const-profile=<file>\n";
    return false;
  }

  FileSharedLockGuard ReadLock(ProfilePath);

  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr = MemoryBuffer::getFile(ProfilePath);
  if (!BufferOrErr) {
    errs() << "Error: Cannot open profile log file: " << ProfilePath << "\n";
    return false;
  }

  StringRef Content = BufferOrErr.get()->getBuffer();
  Expected<json::Value> Parsed = json::parse(Content);
  if (!Parsed) {
    errs() << "Error: Failed to parse JSON profile log: " << toString(Parsed.takeError()) << "\n";
    return false;
  }

  json::Object *RootObj = Parsed->getAsObject();
  if (!RootObj) {
    errs() << "Error: JSON root is not an object\n";
    return false;
  }

  // 仅解析详细 launches（kernels）用于统计
  if (auto KernelsArr = RootObj->getArray("kernels")) {
    for (json::Value &KVal : *KernelsArr) {
      json::Object *KObj = KVal.getAsObject();
      if (!KObj) continue;
      auto NameVal = KObj->getString("device_side_name");
      if (!NameVal)
        NameVal = KObj->getString("name");
      if (!NameVal) continue;
      KernelConstProfile &KP = KernelProfiles[NameVal->str()];
      KP.name = NameVal->str();

      // 记录参数索引到名字的映射（用于可读输出）
      if (auto ParamsArr = KObj->getArray("params")) {
        for (json::Value &PVal : *ParamsArr) {
          json::Object *PObj = PVal.getAsObject();
          if (!PObj) continue;
          if (auto IndexVal = PObj->get("index")) {
            if (auto IndexInt = IndexVal->getAsInteger()) {
              if (auto N = PObj->getString("name")) {
                KP.argIndexToName[(unsigned)*IndexInt] = N->str();
              }
            }
          }
        }
      }

      // 收集本次 launch 的参数值（扁平路径），用于后续频率统计
      std::map<std::string, std::string> LaunchArgMap;
      if (auto ParamsArr = KObj->getArray("params")) {
        for (json::Value &PVal : *ParamsArr) {
          if (auto *PObj = PVal.getAsObject()) {
            if (auto Index = PObj->getInteger("index")) {
              collectParamValues(*PObj, *Index, LaunchArgMap);
            }
          }
        }
      }

      // Add grid and block dimensions from profile
      auto parseDim = [&](const char* dimName, unsigned baseIndex) {
          if (auto DimArr = KObj->getArray(dimName)) {
              if (DimArr->size() > 0) {
                  if(auto V = (*DimArr)[0].getAsInteger()) LaunchArgMap[std::to_string(baseIndex) + ".0"] = std::to_string(*V);
                  else if (auto V = (*DimArr)[0].getAsNumber()) LaunchArgMap[std::to_string(baseIndex) + ".0"] = std::to_string((long long)std::llround(*V));
              }
              if (DimArr->size() > 1) {
                  if(auto V = (*DimArr)[1].getAsInteger()) LaunchArgMap[std::to_string(baseIndex - 1) + ".0"] = std::to_string(*V);
                  else if (auto V = (*DimArr)[1].getAsNumber()) LaunchArgMap[std::to_string(baseIndex - 1) + ".0"] = std::to_string((long long)std::llround(*V));
              }
              if (DimArr->size() > 2) {
                  if(auto V = (*DimArr)[2].getAsInteger()) LaunchArgMap[std::to_string(baseIndex - 2) + ".0"] = std::to_string(*V);
                  else if (auto V = (*DimArr)[2].getAsNumber()) LaunchArgMap[std::to_string(baseIndex - 2) + ".0"] = std::to_string((long long)std::llround(*V));
              }
          }
      };

      parseDim("grid", GRID_DIM_X_INDEX);
      parseDim("block", BLOCK_DIM_X_INDEX);

      if (!LaunchArgMap.empty())
        KP.launches.push_back(std::move(LaunchArgMap));
    }
  }

  return true;
}

// Check if a parameter is used in control flow (loops, branches, etc.)
bool CudaKernelConstPass::isUsedInControlFlow(const Function &F, unsigned paramIndex) {
  if (paramIndex >= F.arg_size())
    return false;
  
  return isUsedInLoop(F, paramIndex) || isUsedInBranch(F, paramIndex);
}

// Check if a parameter is used in loop conditions or bounds
bool CudaKernelConstPass::isUsedInLoop(const Function &F, unsigned paramIndex) {
  if (paramIndex >= F.arg_size())
    return false;
  
  const Argument *Arg = F.getArg(paramIndex);
  
  // Check all uses of this argument
  for (const User *U : Arg->users()) {
    if (const Instruction *I = dyn_cast<Instruction>(U)) {
      // Check if used in comparison instructions (common in loop conditions)
      if (isa<ICmpInst>(I) || isa<FCmpInst>(I)) {
        // Check if this comparison is used in a branch instruction
        for (const User *CmpUser : I->users()) {
          if (isa<BranchInst>(CmpUser)) {
            return true;
          }
        }
      }
      
      // Check if used in PHI nodes (common in loop induction variables)
      if (isa<PHINode>(I)) {
        return true;
      }
    }
  }
  
  return false;
}

// Check if a parameter is used in branch conditions
bool CudaKernelConstPass::isUsedInBranch(const Function &F, unsigned paramIndex) {
  if (paramIndex >= F.arg_size())
    return false;
  
  const Argument *Arg = F.getArg(paramIndex);
  
  // Check all uses of this argument
  for (const User *U : Arg->users()) {
    if (const Instruction *I = dyn_cast<Instruction>(U)) {
      // Check if used directly in branch instructions
      if (const BranchInst *BI = dyn_cast<BranchInst>(I)) {
        if (BI->isConditional()) {
          return true;
        }
      }
      
      // Check if used in comparison instructions
      if (isa<ICmpInst>(I) || isa<FCmpInst>(I)) {
        // Check if this comparison is used in a branch instruction
        for (const User *CmpUser : I->users()) {
          if (const BranchInst *BI = dyn_cast<BranchInst>(CmpUser)) {
            if (BI->isConditional()) {
              return true;
            }
          }
        }
      }
    }
  }
  
  return false;
}

// Check if this kernel should be optimized
bool CudaKernelConstPass::shouldOptimizeKernel(const Function &F) {
  auto It = KernelProfiles.find(F.getName().str());
  if (It == KernelProfiles.end())
    return false;
  
  const KernelConstProfile &Profile = It->second;
  
  // Check if any common scalar is used in control flow
  for (const ScalarConstInfo &SCI : Profile.commonScalars) {
    if (isUsedInControlFlow(F, SCI.index)) {
      return true;
    }
  }
  
  // TODO: Add dimension checking logic if needed
  
  return false;
}

// Compute best parameter combinations per kernel using analysis weights and profile launches
void CudaKernelConstPass::computeBestSelections(Module &M, ModuleAnalysisManager &AM) {
  SelectedScalarsByKernel.clear();
  SelectedParamsByKernel.clear();

  // Access FunctionAnalysisManager from ModuleAnalysisManager
  auto &FAMProxy = AM.getResult<FunctionAnalysisManagerModuleProxy>(M);
  FunctionAnalysisManager &FAM = FAMProxy.getManager();

  for (Function &F : M) {
    if (!isKernelFunction(F))
      continue;

    auto KIt = KernelProfiles.find(F.getName().str());
    if (KIt == KernelProfiles.end())
      continue;

    const KernelConstProfile &KP = KIt->second;
    if (KP.launches.empty())
      continue;

    // 候选参数来源于分析权重（仅权重大于0的参与）。
    auto AR = FAM.getResult<CudaKernelAnalysis>(F);

    struct CandidateParam { unsigned Index; SmallVector<unsigned, 4> Indices; std::string Path; double Weight; };
    std::vector<CandidateParam> Candidates;
    for (const auto &Entry : AR.ScalarWeights) {
      const ParameterInfo &PI = Entry.first;
      int Weight = Entry.second;
      if (Weight <= 0)
        continue;
      unsigned ArgIndex = PI.getArgIndex();
      std::string Path = std::to_string(PI.Indices[0]);
      for (size_t ii = 1; ii < PI.Indices.size(); ++ii)
        Path += "." + std::to_string(PI.Indices[ii]);
      Candidates.push_back({ArgIndex, PI.Indices, std::move(Path), static_cast<double>(Weight)});
    }

    if (Candidates.empty())
      continue;

    // 枚举子集，按联合众数计算 Ratio 与得分，选最佳子集。
    double BestScore = -1.0;
    unsigned BestMask = 0;
    SmallVector<unsigned, 8> BestSubIdx;
    SmallVector<SmallVector<unsigned, 4>, 8> BestSubIndicesPathVec;
    SmallVector<std::string, 8> BestValues; // 与 BestSubIdx 一一对应
    const unsigned Num = Candidates.size();

    for (unsigned Mask = 1; Mask < (1u << Num); ++Mask) {
      // 收集子集信息
      SmallVector<unsigned, 8> SubIdx;
      SmallVector<SmallVector<unsigned, 4>, 8> SubIndicesPathVec;
      SmallVector<std::string, 8> SubPaths;
      SmallVector<double, 8> SubWeights;
      SmallVector<unsigned, 8> SubCandIdx; // 原 Candidates 下标
      for (unsigned i = 0; i < Num; ++i) {
        if (Mask & (1u << i)) {
          SubIdx.push_back(Candidates[i].Index);
          SubIndicesPathVec.push_back(Candidates[i].Indices);
          SubPaths.push_back(Candidates[i].Path);
          SubWeights.push_back(Candidates[i].Weight);
          SubCandIdx.push_back(i);
        }
      }

      // 统计联合众数
      llvm::StringMap<size_t> ComboCount;
      llvm::StringMap<SmallVector<std::string, 8>> ComboValues;
      size_t MaxCount = 0;
      std::string BestKey;

      for (const auto &L : KP.launches) {
        // if (CudaConstDebug) {
        //   dbgs() << "[CudaKernelConst] Launch profile map:\n";
        //   for (auto const& [PathKey, Val] : L) {
        //     dbgs() << "  - " << PathKey << " = " << Val << "\n";
        //   }
        // }

        SmallVector<std::string, 8> Values;
        bool Ok = true;
        for (const auto &Path : SubPaths) {
          // if (CudaConstDebug) {
          //   dbgs() << "[CudaKernelConst] Looking for path: " << Path << "\n";
          // }
          auto It = L.find(Path);
          if (It == L.end()) { Ok = false; break; }
          if (!isSupportedScalarValue(It->second)) { Ok = false; break; }
          Values.push_back(It->second);
        }
        if (!Ok) continue;
        // 组装 key（使用分隔符 '\x1F' 避免冲突）
        std::string Key;
        for (size_t k = 0; k < Values.size(); ++k) {
          if (k) Key.push_back('\x1F');
          Key += Values[k];
        }
        size_t NewCount = ++ComboCount[Key];
        if (NewCount > MaxCount) {
          MaxCount = NewCount;
          BestKey = Key;
          ComboValues[Key] = std::move(Values);
        }
      }

      double Ratio = KP.launches.empty() ? 0.0
                                         : static_cast<double>(MaxCount) / static_cast<double>(KP.launches.size());

      double Score = 0.0;
      for (double W : SubWeights) Score += W * Ratio;

      auto PrintSubset = [&](raw_ostream &OS) {
        OS << "[CudaKernelConst] kernel=" << F.getName()
           << " subset_mask=0x" << Twine::utohexstr(Mask)
           << " ratio=" << Ratio << " score=" << Score
           << " combo_count=" << MaxCount << "/" << KP.launches.size();
        // 打印本子集的联合众数取值
        if (!BestKey.empty() && ComboValues.count(BestKey)) {
          const auto &Vals = ComboValues[BestKey];
          OS << " combo={";
          for (size_t i = 0; i < Vals.size(); ++i) {
            if (i) OS << ", ";
            OS << "arg" << SubIdx[i] << "(path=" << SubPaths[i] << ")=" << Vals[i];
          }
          OS << "}";
        }
        OS << "\n";
      };
      // if (CudaConstDebug) { PrintSubset(dbgs()); } else { LLVM_DEBUG(PrintSubset(dbgs())); }

      if (Score > BestScore ||
          (Score == BestScore && BestMask != 0 &&
           __builtin_popcount(BestMask) < __builtin_popcount(Mask))) {
        BestScore = Score;
        BestMask = Mask;
        BestSubIdx = std::move(SubIdx);
        BestSubIndicesPathVec = std::move(SubIndicesPathVec);
        BestValues = ComboValues.count(BestKey) ? ComboValues[BestKey] : SmallVector<std::string, 8>{};
      }
    }

    if (BestMask == 0 || BestValues.empty())
      continue;

    // 产出 Selected（scalar 列表）与 SelectedParams（带 indices 路径），ratio 可单独按单参统计或使用子集 Ratio。
    std::vector<ScalarConstInfo> Selected;
    std::vector<SelectedParamRecord> SelectedParams;

    // 为显示/元数据需要，计算每个单参所选值的单独比例
    DenseMap<unsigned, double> SingleRatio;
    for (size_t i = 0; i < BestSubIdx.size(); ++i) {
      const unsigned ArgIndex = BestSubIdx[i];
      const std::string &Path = [&]() {
        std::string P = std::to_string((int)BestSubIndicesPathVec[i][0]);
        for (size_t ii = 1; ii < BestSubIndicesPathVec[i].size(); ++ii) P += "." + std::to_string((int)BestSubIndicesPathVec[i][ii]);
        return P;
      }();
      const std::string &Val = BestValues[i];
      size_t Cnt = 0;
      for (const auto &L : KP.launches) {
        auto It = L.find(Path);
        if (It != L.end() && It->second == Val)
          ++Cnt;
      }
      double R = KP.launches.empty() ? 0.0 : static_cast<double>(Cnt) / static_cast<double>(KP.launches.size());
      SingleRatio[ArgIndex] = R;
    }

    for (size_t i = 0; i < BestSubIdx.size(); ++i) {
      if (!isSupportedScalarValue(BestValues[i])) continue;
      ScalarConstInfo S{};
      S.index = BestSubIdx[i];
      S.value = BestValues[i];
      S.ratio = SingleRatio.lookup(S.index);
      Selected.push_back(S);

      SelectedParamRecord SP;
      SP.Indices.assign(BestSubIndicesPathVec[i].begin(), BestSubIndicesPathVec[i].end());
      SP.Value = BestValues[i];
      SP.Ratio = S.ratio;
      if (!SP.Indices.empty() && SP.Indices[0] >= SPECIAL_INDEX_BASE - 5) {
        SP.Type = "i32";
      } else if (Type *SelTy = inferSelectedParamType(F, SP.Indices)) {
        SP.Type = getScalarTypeString(SelTy);
      } else {
        SP.Type = "unknown";
      }
      SelectedParams.push_back(std::move(SP));
    }

    // 打印最终最佳子集的详细选择
    auto PrintBest = [&](raw_ostream &OS) {
      OS << "[CudaKernelConst] kernel=" << F.getName()
         << " best_mask=0x" << Twine::utohexstr(BestMask)
         << " best_score=" << BestScore << " selected={";
      for (size_t i = 0; i < BestSubIdx.size(); ++i) {
        if (i) OS << ", ";
        // 重建 path 输出
        std::string P = std::to_string((int)BestSubIndicesPathVec[i][0]);
        for (size_t ii = 1; ii < BestSubIndicesPathVec[i].size(); ++ii) P += "." + std::to_string((int)BestSubIndicesPathVec[i][ii]);
        OS << "arg" << BestSubIdx[i] << "(path=" << P << ")=" << BestValues[i]
           << " (single_ratio=" << Selected[i].ratio << ")";
      }
      OS << "}" << "\n";
    };
    if (CudaConstDebug) { PrintBest(dbgs()); } else { LLVM_DEBUG(PrintBest(dbgs())); }

    if (!Selected.empty()) {
      SelectedScalarsByKernel[F.getName().str()] = std::move(Selected);
      SelectedParamsByKernel[F.getName().str()] = std::move(SelectedParams);
    }
  }
}

// Create a constant value from string representation
Value *CudaKernelConstPass::getConstantValue(const std::string &valueStr, Type *type) {
  if (type->isIntegerTy()) {
    llvm::StringRef S(valueStr);
    long long iv = 0;
    if (!S.getAsInteger(10, iv)) {
      return ConstantInt::get(type, (uint64_t)iv);
    }
    char *end = nullptr;
    double dv = std::strtod(valueStr.c_str(), &end);
    if (end && *end == '\0') {
      long long iv2 = (long long)std::llround(dv);
      return ConstantInt::get(type, (uint64_t)iv2);
    }
    return nullptr;
  } else if (type->isFloatingPointTy()) {
    char *end = nullptr;
    double dv = std::strtod(valueStr.c_str(), &end);
    if (!(end && *end == '\0')) return nullptr;
    return ConstantFP::get(type, dv);
  }
  
  return nullptr;
}

// Helper: stringify scalar type
static std::string getScalarTypeString(Type *Ty) {
  if (!Ty) return "unknown";
  if (Ty->isIntegerTy()) return ("i" + Twine(Ty->getIntegerBitWidth())).str();
  if (Ty->isHalfTy()) return "half";
  if (Ty->isBFloatTy()) return "bfloat16";
  if (Ty->isFloatTy()) return "float";
  if (Ty->isDoubleTy()) return "double";
  if (Ty->isFP128Ty()) return "fp128";
  if (Ty->isPointerTy()) return "pointer";
  return "unknown";
}

// Helper: find a load type from uses of an argument at a concrete byte offset
static Type *findLoadTypeForArgOffset(const DataLayout &DL, Argument *Arg, uint64_t Offset) {
  if (!Arg) return nullptr;
  for (User *U : Arg->users()) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
      APInt GEP_Offset(DL.getPointerSizeInBits(GEP->getPointerAddressSpace()), 0);
      if (!GEP->accumulateConstantOffset(DL, GEP_Offset)) continue;
      if (GEP_Offset.getZExtValue() != Offset) continue;
      SmallVector<Value *, 8> Worklist;
      SmallPtrSet<Value *, 8> Visited;
      Worklist.push_back(GEP);
      while (!Worklist.empty()) {
        Value *V = Worklist.pop_back_val();
        if (!Visited.insert(V).second) continue;
        for (User *GU : V->users()) {
          if (auto *BC = dyn_cast<BitCastInst>(GU)) { Worklist.push_back(BC); continue; }
          if (auto *LI = dyn_cast<LoadInst>(GU)) { return LI->getType(); }
        }
      }
    }
  }
  // Fallback: if Arg itself is scalar and offset is zero
  if (Offset == 0 && Arg->getType()->isSingleValueType()) return Arg->getType();
  return nullptr;
}

// Helper: infer selected parameter type based on indices path [ArgIndex(,Offset)]
static Type *inferSelectedParamType(Function &F, ArrayRef<unsigned> Indices) {
  if (Indices.empty()) return nullptr;
  unsigned ArgIndex = Indices[0];

  // Handle special indices for grid/block dimensions
  if (ArgIndex >= SPECIAL_INDEX_BASE) {
      // These are always i32
      return Type::getInt32Ty(F.getContext());
  }

  uint64_t Offset = 0;
  if (Indices.size() >= 2) Offset = Indices[1];
  if (ArgIndex >= F.arg_size()) return nullptr;
  Argument *Arg = F.getArg(ArgIndex);
  const DataLayout &DL = F.getParent()->getDataLayout();
  // If offset==0 and argument is a scalar, return it directly
  if (Offset == 0 && (Arg->getType()->isIntegerTy() || Arg->getType()->isFloatingPointTy()))
    return Arg->getType();
  // Otherwise, attempt to find a load type at the offset
  return findLoadTypeForArgOffset(DL, Arg, Offset);
}
// Replace all uses of a parameter with a constant value
void CudaKernelConstPass::replaceUsesWithConstant(Function &F, unsigned paramIndex, 
                                                  Value *constantValue) {
  if (paramIndex >= F.arg_size())
    return;
  
  Argument *Arg = F.getArg(paramIndex);
  Arg->replaceAllUsesWith(constantValue);
}

void CudaKernelConstPass::performConstantPropagation(Function &F,
                                                     const KernelConstProfile &Profile,
                                                     ArrayRef<SelectedParamRecord> SPs) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  Module *M = F.getParent();

  for (const SelectedParamRecord &SP : SPs) {
    // Normalize indices: accept [arg] as [arg,0]
    unsigned ArgIndex = 0;
    uint64_t Offset = 0;
    if (SP.Indices.empty()) continue;
    if (SP.Indices.size() == 1) {
      ArgIndex = SP.Indices[0];
      Offset = 0;
    } else {
      ArgIndex = SP.Indices[0];
      Offset = SP.Indices[1];
    }

    // Handle special grid/block dim parameters
    if (ArgIndex >= SPECIAL_INDEX_BASE - 5) {
      Type *Ty = Type::getInt32Ty(F.getContext());
      Value *ConstVal = getConstantValue(SP.Value, Ty);
      if (!ConstVal) continue;

      StringRef IntrinsicName;
      switch (ArgIndex) {
        case GRID_DIM_X_INDEX: IntrinsicName = "llvm.nvvm.read.ptx.sreg.nctaid.x"; break;
        case GRID_DIM_Y_INDEX: IntrinsicName = "llvm.nvvm.read.ptx.sreg.nctaid.y"; break;
        case GRID_DIM_Z_INDEX: IntrinsicName = "llvm.nvvm.read.ptx.sreg.nctaid.z"; break;
        case BLOCK_DIM_X_INDEX: IntrinsicName = "llvm.nvvm.read.ptx.sreg.ntid.x"; break;
        case BLOCK_DIM_Y_INDEX: IntrinsicName = "llvm.nvvm.read.ptx.sreg.ntid.y"; break;
        case BLOCK_DIM_Z_INDEX: IntrinsicName = "llvm.nvvm.read.ptx.sreg.ntid.z"; break;
      }

      if (!IntrinsicName.empty()) {
        Function *Intrinsic = M->getFunction(IntrinsicName);
        if (Intrinsic) {
          SmallVector<Instruction*, 8> ToErase;
          for (Instruction &I : instructions(F)) {
            if (auto *CI = dyn_cast<CallInst>(&I)) {
              if (CI->getCalledFunction() == Intrinsic) {
                CI->replaceAllUsesWith(ConstVal);
                ToErase.push_back(CI);
              }
            }
          }
          for (Instruction *I : ToErase) {
            I->eraseFromParent();
          }
        }
      }
      continue;
    }
 
    if (ArgIndex >= F.arg_size()) continue;
    Argument *Arg = F.getArg(ArgIndex);

    // If targeting top-level parameter offset==0, attempt direct replacement.
    if (Offset == 0) {
      // Only handle scalar types here
      Type *ATy = Arg->getType();
      Value *ConstVal = getConstantValue(SP.Value, ATy);
      if (ConstVal && ConstVal->getType() == ATy) {
        replaceUsesWithConstant(F, ArgIndex, ConstVal);
        continue;
      }
      // If the argument is pointer or non-scalar, fall through to look for loads of GEP with offset 0
    }

    // Find the GEP with the correct offset and replace its loading users.
    for (User *U : Arg->users()) {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        APInt GEP_Offset(DL.getPointerSizeInBits(GEP->getPointerAddressSpace()), 0);
        if (GEP->accumulateConstantOffset(DL, GEP_Offset)) {
          if (GEP_Offset.getZExtValue() == Offset) {
            // Found the right GEP. Trace through bitcasts to find all loads
            // that use this GEP and replace them.
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
                  Type *LoadTy = LI->getType();
                  Value *ConstVal = getConstantValue(SP.Value, LoadTy);
                  if (ConstVal && ConstVal->getType() == LoadTy) {
                    LI->replaceAllUsesWith(ConstVal);
                  }
                }
              }
            }
            // We have handled all users of the GEP for this offset. We
            // continue here to handle other GEPs with the same offset.
          }
        }
      }
    }
  }
}

// Clone kernel with constant propagation
Function *CudaKernelConstPass::cloneKernelWithConstProp(Function &OrigF, 
                                                       const KernelConstProfile &Profile,
                                                       ArrayRef<SelectedParamRecord> SPs) {
  ValueToValueMapTy VMap;
  Function *ClonedF = CloneFunction(&OrigF, VMap);
  
  // Set new function name for the cloned variant
  std::string NewName = OrigF.getName().str() + "_const";
  ClonedF->setName(NewName);
  
  // Perform constant propagation on the cloned kernel
  performConstantPropagation(*ClonedF, Profile, SPs);
  
  // Copy the existing nvvm.annotations entry so the clone is still a kernel
  Module *M = OrigF.getParent();
  NamedMDNode *NMD = M->getOrInsertNamedMetadata("nvvm.annotations");
  
  for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i) {
    const MDNode *MD = NMD->getOperand(i);
    if (MD->getNumOperands() >= 3) {
      if (auto *FMD = mdconst::dyn_extract_or_null<Function>(MD->getOperand(0))) {
        if (FMD == &OrigF) {
          if (auto *Str = dyn_cast<MDString>(MD->getOperand(1))) {
            if (Str->getString() == "kernel") {
              LLVMContext &Ctx = M->getContext();
              Metadata *MDVals[] = {
                ValueAsMetadata::get(ClonedF),
                MDString::get(Ctx, "kernel"),
                MD->getOperand(2)
              };
              MDNode *NewMD = MDNode::get(Ctx, MDVals);
              NMD->addOperand(NewMD);
              break;
            }
          }
        }
      }
    }
  }
  
  return ClonedF;
}

PreservedAnalyses CudaKernelConstPass::run(Module &M, ModuleAnalysisManager &AM) {

  if (!parseProfileLog()) {
    errs() << "Warning: Failed to parse profile log, skipping constant propagation optimization\n";
    return PreservedAnalyses::all();
  }

  // Compute best selections using analysis weights and profile ratios
  computeBestSelections(M, AM);

  bool Changed = false;
  std::vector<Function *> KernelsToProcess;
  
  // Collect all kernel functions in the module
  for (Function &F : M) {
    if (isKernelFunction(F)) {
      KernelsToProcess.push_back(&F);
    }
  }
  
  // Process each kernel function based on selections
  for (Function *F : KernelsToProcess) {
    auto SelIt = SelectedScalarsByKernel.find(F->getName().str());
    if (SelIt == SelectedScalarsByKernel.end() || SelIt->second.empty())
      continue;

    // Create a filtered profile consisting only of the selected scalars
    KernelConstProfile Filtered;
    Filtered.name = F->getName().str();
    Filtered.commonScalars = SelIt->second;

    auto SPIt = SelectedParamsByKernel.find(F->getName().str());
    SmallVector<SelectedParamRecord, 4> SPs;
    if (SPIt != SelectedParamsByKernel.end()) SPs.assign(SPIt->second.begin(), SPIt->second.end());

    Function *ClonedF = cloneKernelWithConstProp(*F, Filtered, SPs);
    (void)ClonedF; // silence unused warning in builds without further use
    Changed = true;

    LLVM_DEBUG(dbgs() << "Cloned kernel " << F->getName() 
                      << " to " << ClonedF->getName() 
                      << " with constant propagation for " 
                      << Filtered.commonScalars.size() 
                      << " selected scalar parameters\n");
  }
  
  if (!CudaConstSelectedOut.empty()) {
    dumpSelectionsToJson(SelectedParamsByKernel, CudaConstSelectedOut);
  }
  // If not explicitly writing separate selections, merge into original profile
  else if (!CudaConstProfilePath.empty()) {
    mergeSelectionsIntoProfile(SelectedParamsByKernel, CudaConstProfilePath);
  } else if (const char *Env = std::getenv("CUDA_KERNEL_PROFILE")) {
    mergeSelectionsIntoProfile(SelectedParamsByKernel, std::string(Env));
  }
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
} 

// Optionally dump selections to JSON for host-side
static void dumpSelectionsToJson(const std::map<std::string, std::vector<SelectedParamRecord>> &Selected,
                                 const std::string &OutPath) {
  if (OutPath.empty()) return;

  FileLockGuard Lock(OutPath);
  if (!Lock.acquired()) {
    errs() << "Warning: cannot acquire lock for '" << OutPath << "', skipping write.\n";
    return;
  }

  // Load existing file (if any) and build a name->items map
  std::map<std::string, std::vector<SelectedParamRecord>> Existing;
  if (auto BufOrErr = llvm::MemoryBuffer::getFile(OutPath)) {
    if (auto Parsed = llvm::json::parse(BufOrErr.get()->getBuffer())) {
      if (auto *RootObj = Parsed->getAsObject()) {
        // Accept both object and array forms
        if (auto *SelObj = RootObj->getObject("cuda_const_selected")) {
          for (auto &KV : *SelObj) {
            std::string KName = KV.first.str();
            if (auto *Arr = KV.second.getAsArray()) {
              for (const auto &V : *Arr) if (auto *O = V.getAsObject()) {
                SelectedParamRecord SP;
                if (auto *Idxs = O->getArray("indices")) {
                  for (const auto &X : *Idxs) if (auto I = X.getAsInteger()) SP.Indices.push_back((unsigned)*I);
                }
                if (auto S = O->getString("value")) SP.Value = S->str();
                if (auto R = O->getNumber("ratio")) SP.Ratio = *R;
                if (auto T = O->getString("type")) SP.Type = T->str();
                if (!SP.Indices.empty()) Existing[KName].push_back(std::move(SP));
              }
            }
          }
        } else if (auto *SelArr = RootObj->getArray("cuda_const_selected")) {
          for (const auto &Elem : *SelArr) if (auto *Obj = Elem.getAsObject()) {
            std::string KName;
            if (auto KS = Obj->getString("name")) KName = KS->str();
            else if (auto KS2 = Obj->getString("kernel")) KName = KS2->str();
            if (KName.empty()) continue;
            if (auto *Arr = Obj->getArray("selected")) {
              for (const auto &V : *Arr) if (auto *O = V.getAsObject()) {
                SelectedParamRecord SP;
                if (auto *Idxs = O->getArray("indices")) {
                  for (const auto &X : *Idxs) if (auto I = X.getAsInteger()) SP.Indices.push_back((unsigned)*I);
                }
                if (auto S = O->getString("value")) SP.Value = S->str();
                if (auto R = O->getNumber("ratio")) SP.Ratio = *R;
                if (auto T = O->getString("type")) SP.Type = T->str();
                if (!SP.Indices.empty()) Existing[KName].push_back(std::move(SP));
              }
            }
          }
        }
      }
    }
  }

  auto indicesEqual = [](const std::vector<unsigned> &A, const std::vector<unsigned> &B) -> bool {
    if (A.size() != B.size()) return false; for (size_t i = 0; i < A.size(); ++i) if (A[i] != B[i]) return false; return true;
  };

  // Merge new selections into Existing (dedupe by indices; update value/ratio)
  for (const auto &KV : Selected) {
    const std::string &KName = KV.first; const auto &Vec = KV.second;
    auto &Dst = Existing[KName];
    for (const auto &SP : Vec) {
      bool Replaced = false;
      for (auto &Old : Dst) {
        if (indicesEqual(Old.Indices, SP.Indices)) { Old.Value = SP.Value; Old.Ratio = SP.Ratio; if (!SP.Type.empty()) Old.Type = SP.Type; Replaced = true; break; }
      }
      if (!Replaced) Dst.push_back(SP);
    }
  }

  // Serialize back as array form
  llvm::json::Array Kernels;
  for (const auto &KV : Existing) {
    const std::string &KernelName = KV.first;
    const auto &Params = KV.second;
    llvm::json::Object KObj; KObj["name"] = KernelName; llvm::json::Array Items;
    for (const auto &SP : Params) {
      llvm::json::Object Item; llvm::json::Array Idxs;
      for (unsigned idx : SP.Indices) Idxs.push_back((int64_t)idx);
      Item["indices"] = std::move(Idxs); Item["value"] = SP.Value; Item["ratio"] = SP.Ratio; Item["type"] = (SP.Type.empty() ? std::string("unknown") : SP.Type); Items.push_back(std::move(Item));
    }
    KObj["selected"] = std::move(Items); Kernels.push_back(std::move(KObj));
  }

  llvm::json::Object Root; Root["cuda_const_selected"] = std::move(Kernels);
  std::error_code EC; llvm::raw_fd_ostream OS(OutPath, EC);
  if (EC) { errs() << "Failed to write cuda-kernel-const-selected-out to '" << OutPath << "': " << EC.message() << "\n"; return; }
  OS << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(Root)));
} 

static void mergeSelectionsIntoProfile(const std::map<std::string, std::vector<SelectedParamRecord>> &Selected,
                                       const std::string &ProfilePath) {
  if (ProfilePath.empty() || Selected.empty()) return;

  FileLockGuard Lock(ProfilePath);
  if (!Lock.acquired()) {
    errs() << "Warning: cannot acquire lock for '" << ProfilePath << "', skipping write.\n";
    return;
  }

  // Load existing
  std::map<std::string, std::vector<SelectedParamRecord>> Existing;
  llvm::json::Object Root;
  if (auto BufOrErr = llvm::MemoryBuffer::getFile(ProfilePath)) {
    llvm::StringRef Content = BufOrErr.get()->getBuffer();
    if (auto Parsed = llvm::json::parse(Content)) {
      if (auto *Obj = Parsed->getAsObject()) {
        Root = *Obj;
        if (auto *SelObj = Root.getObject("cuda_const_selected")) {
          for (auto &KV : *SelObj) {
            std::string KName = KV.first.str();
            if (auto *Arr = KV.second.getAsArray()) {
              for (const auto &V : *Arr) if (auto *O = V.getAsObject()) {
                SelectedParamRecord SP; if (auto *Idxs = O->getArray("indices")) {
                  for (const auto &X : *Idxs) if (auto I = X.getAsInteger()) SP.Indices.push_back((unsigned)*I);
                } else if (auto *PI = O->getArray("param_index")) {
                  for (const auto &X : *PI) if (auto I = X.getAsInteger()) SP.Indices.push_back((unsigned)*I);
                }
                if (auto S = O->getString("value")) SP.Value = S->str();
                if (auto R = O->getNumber("ratio")) SP.Ratio = *R; // optional
                if (auto T = O->getString("type")) SP.Type = T->str();
                if (!SP.Indices.empty()) Existing[KName].push_back(std::move(SP));
              }
            }
          }
        } else if (auto *SelArr = Root.getArray("cuda_const_selected")) {
          for (const auto &Elem : *SelArr) if (auto *Obj2 = Elem.getAsObject()) {
            std::string KName; if (auto KS = Obj2->getString("name")) KName = KS->str(); else if (auto KS2 = Obj2->getString("kernel")) KName = KS2->str();
            if (KName.empty()) continue; if (auto *Arr = Obj2->getArray("selected")) {
              for (const auto &V : *Arr) if (auto *O = V.getAsObject()) {
                SelectedParamRecord SP; if (auto *Idxs = O->getArray("indices")) {
                  for (const auto &X : *Idxs) if (auto I = X.getAsInteger()) SP.Indices.push_back((unsigned)*I);
                } else if (auto *PI = O->getArray("param_index")) {
                  for (const auto &X : *PI) if (auto I = X.getAsInteger()) SP.Indices.push_back((unsigned)*I);
                }
                if (auto S = O->getString("value")) SP.Value = S->str();
                if (auto R = O->getNumber("ratio")) SP.Ratio = *R;
                if (auto T = O->getString("type")) SP.Type = T->str();
                if (!SP.Indices.empty()) Existing[KName].push_back(std::move(SP));
              }
            }
          }
        }
      }
    }
  }

  auto indicesEqual = [](const std::vector<unsigned> &A, const std::vector<unsigned> &B) -> bool {
    if (A.size() != B.size()) return false; for (size_t i = 0; i < A.size(); ++i) if (A[i] != B[i]) return false; return true;
  };

  // Merge incoming
  for (const auto &KV : Selected) {
    auto &Dst = Existing[KV.first];
    for (const auto &SP : KV.second) {
      bool Replaced = false; for (auto &Old : Dst) { if (indicesEqual(Old.Indices, SP.Indices)) { Old.Value = SP.Value; Old.Ratio = SP.Ratio; if (!SP.Type.empty()) Old.Type = SP.Type; Replaced = true; break; } }
      if (!Replaced) Dst.push_back(SP);
    }
  }

  // Serialize back as array form
  llvm::json::Array Kernels;
  for (const auto &KV : Existing) {
    llvm::json::Object KObj; KObj["name"] = KV.first; llvm::json::Array Items;
    for (const auto &SP : KV.second) { llvm::json::Object Item; llvm::json::Array Idxs; for (unsigned idx : SP.Indices) Idxs.push_back((int64_t)idx);
      Item["indices"] = std::move(Idxs); Item["value"] = SP.Value; Item["ratio"] = SP.Ratio; Item["type"] = (SP.Type.empty() ? std::string("unknown") : SP.Type); Items.push_back(std::move(Item)); }
    KObj["selected"] = std::move(Items); Kernels.push_back(std::move(KObj));
  }
  Root["cuda_const_selected"] = std::move(Kernels);
  std::error_code EC; llvm::raw_fd_ostream OS(ProfilePath, EC);
  if (EC) { errs() << "Failed to update profile file '" << ProfilePath << "' with selections: " << EC.message() << "\n"; return; }
  OS << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(Root)));
}