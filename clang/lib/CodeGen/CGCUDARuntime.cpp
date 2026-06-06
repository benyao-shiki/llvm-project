//===----- CGCUDARuntime.cpp - Interface to CUDA Runtimes -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for CUDA code generation.  Concrete
// subclasses of this implement code generation for specific CUDA
// runtime libraries.
//
//===----------------------------------------------------------------------===//

#include "CGCUDARuntime.h"
#include "CGCall.h"
#include "CodeGenFunction.h"
#include "clang/AST/ExprCXX.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include <cstdlib>
#include <vector>
#include <cstring>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

using namespace clang;
using namespace CodeGen;

//===----------------------------------------------------------------------===//
// Helper utilities for CUDA kernel profile parsing (shared with LLVM pass)
//===----------------------------------------------------------------------===//

namespace {
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
struct KernelProfileInfo {
  std::vector<unsigned> TargetPointerIndices; ///< Pointer parameter indices
};

struct ScalarConstInfo {
  unsigned index;     // Argument index (0-based)
  std::string value;  // Common constant value as string
  double ratio;       // Frequency ratio (0.0 to 1.0)
  std::vector<unsigned> indices; // Full indices path (e.g., [arg, member...])
  std::string type;
};

struct KernelConstProfileInfo {
  std::vector<ScalarConstInfo> commonScalars;
};

// Special indices for implicit CUDA parameters
// These must be kept in sync with the LLVM pass CudaKernelConst.cpp
static constexpr unsigned SPECIAL_INDEX_BASE = 0xE0000000;
static constexpr unsigned GRID_DIM_X_INDEX  = SPECIAL_INDEX_BASE;
static constexpr unsigned GRID_DIM_Y_INDEX  = SPECIAL_INDEX_BASE - 1;
static constexpr unsigned GRID_DIM_Z_INDEX  = SPECIAL_INDEX_BASE - 2;
static constexpr unsigned BLOCK_DIM_X_INDEX = SPECIAL_INDEX_BASE - 3;
static constexpr unsigned BLOCK_DIM_Y_INDEX = SPECIAL_INDEX_BASE - 4;
static constexpr unsigned BLOCK_DIM_Z_INDEX = SPECIAL_INDEX_BASE - 5;

static llvm::StringMap<KernelProfileInfo> KernelProfileMap;
static llvm::StringMap<KernelConstProfileInfo> KernelConstProfileMap;
static llvm::StringMap<std::vector<std::vector<unsigned>>> KernelNoaliasProfileMap; // kernel -> list of indices paths
static bool KernelProfileLoaded = false;

static void loadCudaKernelProfile(const CodeGenModule &CGM) {
  if (KernelProfileLoaded)
    return;
  KernelProfileLoaded = true;
  
    std::string ProfilePath;
  // 1) Environment variable takes priority.
  if (const char *Env = std::getenv("CUDA_KERNEL_PROFILE"))
    ProfilePath = Env;

  // 2) Fallback: scan backend options for "-cuda-kernel-profile=" or "-cuda-kernel-const-profile=".
  if (ProfilePath.empty()) {
    for (const std::string &Opt : CGM.getCodeGenOpts().CommandLineArgs) {
      llvm::StringRef S(Opt);
      if (S.consume_front("-cuda-kernel-profile=")) {
        ProfilePath = S.str();
        break;
      } else if (S.consume_front("-cuda-kernel-const-profile=")) {
        ProfilePath = S.str();
        break;
      }
    }
  }
  if (ProfilePath.empty())
    return; // Silently give up – no profile info available.

  FileSharedLockGuard ReadLock(ProfilePath);

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> BufferOrErr =
      llvm::MemoryBuffer::getFile(ProfilePath);
  if (!BufferOrErr)
    return;

  llvm::StringRef Content = BufferOrErr.get()->getBuffer();
  auto Parsed = llvm::json::parse(Content);
  if (!Parsed)
    return;

  llvm::json::Object *RootObj = Parsed->getAsObject();
  if (!RootObj)
    return;

  // 0) Prefer cuda_const_selected if present; it contains final device-side selections.
  {
    auto addConstItem = [&](llvm::StringRef Kernel, std::vector<unsigned> Indices, std::string ValueStr, std::string TypeStr = "") {
      KernelConstProfileInfo &Slot = KernelConstProfileMap[Kernel];
      ScalarConstInfo Item;
      Item.index = Indices.empty() ? 0u : Indices.front();
      Item.value = std::move(ValueStr);
      Item.ratio = 1.0; // final selection – treat as always-on
      Item.indices = std::move(Indices);
      Item.type = std::move(TypeStr);
      Slot.commonScalars.push_back(std::move(Item));
    };

    if (llvm::json::Object *SelObj = RootObj->getObject("cuda_const_selected")) {
      for (auto &KV : *SelObj) {
        llvm::StringRef KernelName = KV.first;
        if (llvm::json::Array *Arr = KV.second.getAsArray()) {
          for (const llvm::json::Value &V : *Arr) {
            if (const llvm::json::Object *O = V.getAsObject()) {
              // Accept shapes: index (int), indices (array), param_index (int|array)
              std::vector<unsigned> IndicesVec;
              if (auto Indices = O->getArray("indices")) {
                for (const auto &X : *Indices) if (auto I = X.getAsInteger()) IndicesVec.push_back((unsigned)*I);
              } else if (auto PIi = O->getInteger("param_index")) {
                IndicesVec.push_back((unsigned)*PIi);
              } else if (auto PIArr = O->getArray("param_index")) {
                for (const auto &X : *PIArr) if (auto I = X.getAsInteger()) IndicesVec.push_back((unsigned)*I);
              } else if (auto Idx = O->getInteger("index")) {
                IndicesVec.push_back((unsigned)*Idx);
              } else {
                continue;
              }
              std::string ValueStr;
 
              if (auto S = O->getString("value")) {
                ValueStr = S->str();
              } else if (auto N = O->get("value")) {
                if (auto Num = N->getAsNumber())
                  ValueStr = llvm::Twine((long long)*Num).str();
                else
                  continue;
              } else {
                continue;
              }
              std::string TypeStr;
              if (auto T = O->getString("type")) {
                TypeStr = T->str();
              }
              addConstItem(KernelName, std::move(IndicesVec), std::move(ValueStr), std::move(TypeStr));
            }
          }
        }
      }
    } else if (const llvm::json::Array *SelArr = RootObj->getArray("cuda_const_selected")) {
      for (const llvm::json::Value &Elem : *SelArr) {
        if (const llvm::json::Object *Obj = Elem.getAsObject()) {
          // Accept shapes like {"name": kernel, "selected": [...]}
          llvm::StringRef KName;
          if (auto KS = Obj->getString("name"))
            KName = *KS;
          else if (auto KS2 = Obj->getString("kernel"))
            KName = *KS2;
          else
            continue;
          if (const llvm::json::Array *Arr = Obj->getArray("selected")) {
            for (const llvm::json::Value &V : *Arr) {
              if (const llvm::json::Object *O = V.getAsObject()) {
                std::vector<unsigned> IndicesVec;
                if (auto Indices = O->getArray("indices")) {
                  for (const auto &X : *Indices) if (auto I = X.getAsInteger()) IndicesVec.push_back((unsigned)*I);
                } else if (auto PIi = O->getInteger("param_index")) {
                  IndicesVec.push_back((unsigned)*PIi);
                } else if (auto PIArr = O->getArray("param_index")) {
                  for (const auto &X : *PIArr) if (auto I = X.getAsInteger()) IndicesVec.push_back((unsigned)*I);
                } else if (auto Idx = O->getInteger("index")) {
                  IndicesVec.push_back((unsigned)*Idx);
                } else {
                  continue;
                }
                std::string ValueStr;
                if (auto S = O->getString("value")) {
                  ValueStr = S->str();
                } else if (auto N = O->get("value")) {
                  if (auto Num = N->getAsNumber())
                    ValueStr = llvm::Twine((long long)*Num).str();
                  else
                    continue;
                } else {
                  continue;
                }
                std::string TypeStr;
                if (auto T = O->getString("type")) {
                  TypeStr = T->str();
                }
                addConstItem(KName, std::move(IndicesVec), std::move(ValueStr), std::move(TypeStr));
              }
            }
          }
        }
      }
    }
  }

  // Legacy 'hot_kernels' parsing removed entirely. Use 'cuda_const_selected' (and in future 'cuda_noalias_selected').

  // Parse cuda_noalias_selected (object or array forms)
  if (llvm::json::Object *NoObj = RootObj->getObject("cuda_noalias_selected")) {
    for (auto &KV : *NoObj) {
      llvm::StringRef KernelName = KV.first;
      if (llvm::json::Array *Arr = KV.second.getAsArray()) {
        for (const llvm::json::Value &V : *Arr) {
          if (const llvm::json::Object *O = V.getAsObject()) {
            std::vector<unsigned> Indices;
            if (auto Idxs = O->getArray("indices")) {
              for (const auto &X : *Idxs) if (auto I = X.getAsInteger()) Indices.push_back((unsigned)*I);
            } else if (auto PI = O->getInteger("param_index")) {
              Indices.push_back((unsigned)*PI);
            } else if (auto PIA = O->getArray("param_index")) {
              for (const auto &X : *PIA) if (auto I = X.getAsInteger()) Indices.push_back((unsigned)*I);
            } else if (auto Idx = O->getInteger("index")) {
              Indices.push_back((unsigned)*Idx);
            }
            if (!Indices.empty()) KernelNoaliasProfileMap[KernelName].push_back(std::move(Indices));
          }
        }
      }
    }
  } else if (const llvm::json::Array *NoArr = RootObj->getArray("cuda_noalias_selected")) {
    for (const llvm::json::Value &Elem : *NoArr) {
      if (const llvm::json::Object *Obj = Elem.getAsObject()) {
        llvm::StringRef KName;
        if (auto KS = Obj->getString("name")) KName = *KS;
        else if (auto KS2 = Obj->getString("kernel")) KName = *KS2;
        else continue;
        if (const llvm::json::Array *Arr = Obj->getArray("selected")) {
          for (const llvm::json::Value &V : *Arr) {
            if (const llvm::json::Object *O = V.getAsObject()) {
              std::vector<unsigned> Indices;
              if (auto Idxs = O->getArray("indices")) {
                for (const auto &X : *Idxs) if (auto I = X.getAsInteger()) Indices.push_back((unsigned)*I);
              } else if (auto PI = O->getInteger("param_index")) {
                Indices.push_back((unsigned)*PI);
              } else if (auto PIA = O->getArray("param_index")) {
                for (const auto &X : *PIA) if (auto I = X.getAsInteger()) Indices.push_back((unsigned)*I);
              } else if (auto Idx = O->getInteger("index")) {
                Indices.push_back((unsigned)*Idx);
              }
              if (!Indices.empty()) KernelNoaliasProfileMap[KName].push_back(std::move(Indices));
            }
          }
        }
      }
    }
  }
}
} // end anonymous namespace

CGCUDARuntime::~CGCUDARuntime() {}

RValue CGCUDARuntime::EmitCUDAKernelCallExpr(CodeGenFunction &CGF,
                                             const CUDAKernelCallExpr *E,
                                             ReturnValueSlot ReturnValue,
                                             llvm::CallBase **CallOrInvoke) {
  llvm::BasicBlock *ConfigOKBlock = CGF.createBasicBlock("kcall.configok");
  llvm::BasicBlock *ContBlock = CGF.createBasicBlock("kcall.end");

  CodeGenFunction::ConditionalEvaluation Eval(CGF);
  CGF.EmitBranchOnBoolExpr(E->getConfig(), ContBlock, ConfigOKBlock,
                           /*TrueCount=*/0);

  Eval.begin(CGF);
  CGF.EmitBlock(ConfigOKBlock);

  const FunctionDecl *FD = dyn_cast<FunctionDecl>(E->getCalleeDecl());
  bool WantConst = FD && CGF.CGM.getCodeGenOpts().CudaKernelConst;
  bool WantNoalias = FD && CGF.CGM.getCodeGenOpts().CudaKernelNoalias;

  if (WantConst || WantNoalias)
    loadCudaKernelProfile(CGF.CGM);

  std::string MangledName;
  std::string SimpleName;
  if (FD) {
    MangledName = CGF.CGM.getMangledName(GlobalDecl(FD)).str();
    SimpleName = FD->getNameAsString();
  }

  auto NameMatches = [&](llvm::StringRef Candidate) -> bool {
    if (!FD)
      return false;
    if (Candidate == MangledName || Candidate == SimpleName ||
        Candidate.ends_with(SimpleName))
      return true;
    llvm::StringRef NameRef(MangledName);
    size_t StubPos = NameRef.find("__device_stub__");
    if (StubPos != llvm::StringRef::npos) {
      llvm::StringRef Suffix = NameRef.substr(StubPos + strlen("__device_stub__"));
      return Candidate.ends_with(Suffix);
    }
    return false;
  };

  const KernelConstProfileInfo *ConstProfile = nullptr;
  if (WantConst) {
    auto It = KernelConstProfileMap.find(MangledName);
    if (It == KernelConstProfileMap.end())
      It = KernelConstProfileMap.find(SimpleName);
    if (It == KernelConstProfileMap.end()) {
      for (auto MapIt = KernelConstProfileMap.begin();
           MapIt != KernelConstProfileMap.end(); ++MapIt) {
        if (NameMatches(MapIt->getKey())) {
          It = MapIt;
          break;
        }
      }
    }
    if (It != KernelConstProfileMap.end() && !It->second.commonScalars.empty())
      ConstProfile = &It->second;
  }

  const std::vector<std::vector<unsigned>> *NoaliasProfile = nullptr;
  if (WantNoalias) {
    auto It = KernelNoaliasProfileMap.find(MangledName);
    if (It == KernelNoaliasProfileMap.end()) {
      for (auto MapIt = KernelNoaliasProfileMap.begin();
           MapIt != KernelNoaliasProfileMap.end(); ++MapIt) {
        if (NameMatches(MapIt->getKey())) {
          It = MapIt;
          break;
        }
      }
    }
    if (It != KernelNoaliasProfileMap.end() && !It->second.empty())
      NoaliasProfile = &It->second;
  }

  auto EmitOriginalCall = [&]() {
    CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
  };

  auto EmitStubCall = [&](llvm::StringRef Suffix) -> bool {
    const Expr *UnderlyingCallee = E->getCallee()->IgnoreParenImpCasts();
    const auto *DRE = dyn_cast<DeclRefExpr>(UnderlyingCallee);
    if (!DRE)
      return false;
    const auto *CalledFD = dyn_cast<FunctionDecl>(DRE->getDecl());
    if (!CalledFD)
      return false;

    std::string BaseStubName =
        CGF.CGM.getMangledName(GlobalDecl(CalledFD, KernelReferenceKind::Stub)).str();
    std::string StubName = BaseStubName + Suffix.str();

    llvm::Function *Stub = CGF.CGM.getModule().getFunction(StubName);
    if (!Stub) {
      const CGFunctionInfo &FI =
          CGF.CGM.getTypes().arrangeGlobalDeclaration(GlobalDecl(CalledFD));
      llvm::FunctionType *FTy = CGF.CGM.getTypes().GetFunctionType(FI);
      Stub = llvm::Function::Create(FTy, llvm::GlobalValue::ExternalLinkage,
                                    StubName, &CGF.CGM.getModule());
    }

    CallArgList Args;
    for (const Expr *Arg : E->arguments())
      Args.add(CGF.EmitAnyExpr(Arg), Arg->getType());

    const CGFunctionInfo &FnInfo = CGF.CGM.getTypes().arrangeFreeFunctionCall(
        Args, CalledFD->getType()->castAs<FunctionType>(), /*ChainCall=*/false);
    CGF.EmitCall(FnInfo, CGCallee::forDirect(Stub), ReturnValueSlot(), Args);
    return true;
  };

  auto EmitSelectedCall = [&](llvm::StringRef Suffix) {
    if (!EmitStubCall(Suffix))
      EmitOriginalCall();
  };

  auto EmitConstCheck = [&](const KernelConstProfileInfo &Profile) -> llvm::Value * {
    llvm::Type *BoolTy = CGF.Builder.getInt1Ty();
    llvm::Type *IntTy = CGF.IntTy;
    llvm::Type *Int64PtrTy = llvm::PointerType::getUnqual(CGF.Builder.getInt64Ty());
    llvm::FunctionType *CheckConstTy = llvm::FunctionType::get(
        BoolTy, {IntTy, Int64PtrTy, Int64PtrTy}, false);
    llvm::FunctionCallee CheckConstFn =
        CGF.CGM.CreateRuntimeFunction(CheckConstTy, "check_const");

    llvm::Constant *NumValues =
        llvm::ConstantInt::get(IntTy, Profile.commonScalars.size());
    llvm::ArrayType *Int64ArrayTy =
        llvm::ArrayType::get(CGF.Builder.getInt64Ty(), Profile.commonScalars.size());
    Address ExpectedArray =
        CGF.CreateTempAlloca(Int64ArrayTy, CGF.getPointerAlign(), "expected_values");
    Address ActualArray =
        CGF.CreateTempAlloca(Int64ArrayTy, CGF.getPointerAlign(), "actual_values");

    for (size_t I = 0; I < Profile.commonScalars.size(); ++I) {
      const std::string &S = Profile.commonScalars[I].value;
      int64_t ExpectedValue = 0;
      if (S.find_first_of(".eE") != std::string::npos)
        ExpectedValue = static_cast<int64_t>(std::stod(S));
      else
        ExpectedValue = std::stoll(S);

      llvm::Value *ExpectedVal =
          llvm::ConstantInt::get(CGF.Builder.getInt64Ty(), ExpectedValue);
      llvm::Value *ExpectedPtr = CGF.Builder.CreateInBoundsGEP(
          Int64ArrayTy, ExpectedArray.emitRawPointer(CGF),
          {CGF.Builder.getInt32(0), CGF.Builder.getInt32(I)});
      Address ExpectedAddr =
          Address(ExpectedPtr, CGF.Builder.getInt64Ty(), CGF.getPointerAlign());
      CGF.Builder.CreateStore(ExpectedVal, ExpectedAddr);
    }

    for (size_t I = 0; I < Profile.commonScalars.size(); ++I) {
      const ScalarConstInfo &SCI = Profile.commonScalars[I];
      const std::vector<unsigned> &Path = SCI.indices;
      if (Path.empty())
        continue;

      unsigned ArgIndex = Path[0];
      llvm::Value *ActualVal = nullptr;

      if (ArgIndex >= SPECIAL_INDEX_BASE - 5) {
        const Expr *DimExpr = nullptr;
        const char *FieldName = nullptr;
        switch (ArgIndex) {
        case GRID_DIM_X_INDEX:  DimExpr = E->getConfig()->getArg(0); FieldName = "x"; break;
        case GRID_DIM_Y_INDEX:  DimExpr = E->getConfig()->getArg(0); FieldName = "y"; break;
        case GRID_DIM_Z_INDEX:  DimExpr = E->getConfig()->getArg(0); FieldName = "z"; break;
        case BLOCK_DIM_X_INDEX: DimExpr = E->getConfig()->getArg(1); FieldName = "x"; break;
        case BLOCK_DIM_Y_INDEX: DimExpr = E->getConfig()->getArg(1); FieldName = "y"; break;
        case BLOCK_DIM_Z_INDEX: DimExpr = E->getConfig()->getArg(1); FieldName = "z"; break;
        }
        if (DimExpr && FieldName) {
          LValue DimLV = CGF.EmitLValue(DimExpr);
          if (const RecordType *RT = DimLV.getType()->getAs<RecordType>()) {
            const RecordDecl *RD = RT->getDecl();
            for (const FieldDecl *Field : RD->fields()) {
              if (Field->getName() == FieldName) {
                LValue FieldLV = CGF.EmitLValueForField(DimLV, Field);
                ActualVal = CGF.EmitLoadOfScalar(FieldLV.getAddress(), false,
                                                  Field->getType(), E->getExprLoc());
                break;
              }
            }
          }
        }
      } else {
        if (ArgIndex >= E->getNumArgs())
          continue;
        const Expr *ArgE = E->getArg(ArgIndex);

        if (Path.size() == 1 || (Path.size() == 2 && Path[1] == 0)) {
          ActualVal = CGF.EmitAnyExpr(ArgE).getScalarVal();
        } else if (Path.size() == 2) {
          uint64_t Offset = Path[1];
          LValue BaseLV = CGF.EmitLValue(ArgE);
          llvm::Value *BasePtr = BaseLV.getPointer(CGF);
          llvm::Value *OffsetVal = llvm::ConstantInt::get(CGF.SizeTy, Offset);
          llvm::Value *GEPPtr = CGF.Builder.CreateInBoundsGEP(
              CGF.Builder.getInt8Ty(), BasePtr, OffsetVal);

          const std::string &TypeStr = SCI.type;
          llvm::Type *LoadTy = nullptr;
          unsigned AlignInBytes = 0;
          if (TypeStr == "long" || TypeStr == "long long" ||
              TypeStr == "unsigned long" || TypeStr == "unsigned long long" ||
              TypeStr == "i64" || TypeStr == "u64") {
            LoadTy = CGF.Builder.getInt64Ty(); AlignInBytes = 8;
          } else if (TypeStr == "int" || TypeStr == "unsigned int" ||
                     TypeStr == "i32" || TypeStr == "u32") {
            LoadTy = CGF.Builder.getInt32Ty(); AlignInBytes = 4;
          } else if (TypeStr == "short" || TypeStr == "unsigned short" ||
                     TypeStr == "i16" || TypeStr == "u16") {
            LoadTy = CGF.Builder.getInt16Ty(); AlignInBytes = 2;
          } else if (TypeStr == "char" || TypeStr == "unsigned char" ||
                     TypeStr == "signed char" || TypeStr == "i8" ||
                     TypeStr == "u8") {
            LoadTy = CGF.Builder.getInt8Ty(); AlignInBytes = 1;
          } else if (TypeStr == "float") {
            LoadTy = CGF.Builder.getFloatTy(); AlignInBytes = 4;
          } else if (TypeStr == "double") {
            LoadTy = CGF.Builder.getDoubleTy(); AlignInBytes = 8;
          }

          if (LoadTy) {
            llvm::Value *CastedPtr = CGF.Builder.CreateBitCast(GEPPtr,
                                                               LoadTy->getPointerTo());
            ActualVal = CGF.Builder.CreateLoad(
                Address(CastedPtr, LoadTy, CharUnits::fromQuantity(AlignInBytes)));
          } else if (const auto *RT = BaseLV.getType()->getAs<RecordType>()) {
            const RecordDecl *RD = RT->getDecl();
            const ASTRecordLayout &Layout = CGF.getContext().getASTRecordLayout(RD);
            const FieldDecl *MatchedField = nullptr;
            unsigned FieldNo = 0;
            for (const FieldDecl *Field : RD->fields()) {
              if (Layout.getFieldOffset(FieldNo) / 8 == Offset) {
                MatchedField = Field;
                break;
              }
              ++FieldNo;
            }
            if (MatchedField) {
              LValue MemberLV = CGF.EmitLValueForField(BaseLV, MatchedField);
              ActualVal = CGF.EmitLoadOfScalar(MemberLV.getAddress(), false,
                                                MatchedField->getType(), E->getExprLoc());
            }
          }

          if (!ActualVal) {
            llvm::Value *CastedPtr = CGF.Builder.CreateBitCast(
                GEPPtr, llvm::PointerType::getUnqual(CGF.Builder.getInt64Ty()));
            ActualVal = CGF.Builder.CreateLoad(
                Address(CastedPtr, CGF.Builder.getInt64Ty(), CGF.getPointerAlign()));
          }
        }
      }

      if (!ActualVal)
        continue;
      if (ActualVal->getType()->isIntegerTy())
        ActualVal = CGF.Builder.CreateSExtOrTrunc(ActualVal, CGF.Builder.getInt64Ty());
      else if (ActualVal->getType()->isFloatingPointTy())
        ActualVal = CGF.Builder.CreateFPToSI(ActualVal, CGF.Builder.getInt64Ty());
      else
        continue;

      llvm::Value *ActualPtr = CGF.Builder.CreateInBoundsGEP(
          Int64ArrayTy, ActualArray.emitRawPointer(CGF),
          {CGF.Builder.getInt32(0), CGF.Builder.getInt32(I)});
      Address ActualAddr =
          Address(ActualPtr, CGF.Builder.getInt64Ty(), CGF.getPointerAlign());
      CGF.Builder.CreateStore(ActualVal, ActualAddr);
    }

    llvm::Value *ExpectedPtr =
        CGF.Builder.CreateBitCast(ExpectedArray.emitRawPointer(CGF), Int64PtrTy);
    llvm::Value *ActualPtr =
        CGF.Builder.CreateBitCast(ActualArray.emitRawPointer(CGF), Int64PtrTy);
    return CGF.EmitRuntimeCallOrInvoke(CheckConstFn,
                                       {NumValues, ExpectedPtr, ActualPtr});
  };

  auto EmitNoaliasCheck = [&](const std::vector<std::vector<unsigned>> &Selections)
      -> llvm::Value * {
    llvm::SmallVector<llvm::Value *, 8> TargetPtrs;
    llvm::SmallVector<llvm::Value *, 8> OtherPtrs;

    auto EmitPtrFromIndices = [&](const std::vector<unsigned> &IdxVec) -> llvm::Value * {
      if (IdxVec.empty())
        return nullptr;
      unsigned ArgIndex = IdxVec[0];
      if (ArgIndex >= E->getNumArgs())
        return nullptr;
      const Expr *ArgE = E->getArg(ArgIndex);

      if (IdxVec.size() == 1 || (IdxVec.size() == 2 && IdxVec[1] == 0)) {
        if (!ArgE->getType()->isPointerType())
          return nullptr;
        return CGF.EmitScalarExpr(ArgE);
      }

      if (IdxVec.size() == 2) {
        uint64_t Offset = IdxVec[1];
        LValue LV = CGF.EmitLValue(ArgE);
        if (const auto *RT = LV.getType()->getAs<RecordType>()) {
          const RecordDecl *RD = RT->getDecl();
          const ASTRecordLayout &Layout = CGF.getContext().getASTRecordLayout(RD);
          const FieldDecl *MatchedField = nullptr;
          unsigned FieldNo = 0;
          for (const FieldDecl *Field : RD->fields()) {
            if (Layout.getFieldOffset(FieldNo) / 8 == Offset) {
              MatchedField = Field;
              break;
            }
            ++FieldNo;
          }
          if (MatchedField && MatchedField->getType()->isPointerType()) {
            LValue MemberLV = CGF.EmitLValueForField(LV, MatchedField);
            return CGF.EmitLoadOfScalar(MemberLV.getAddress(), false,
                                        MatchedField->getType(), E->getExprLoc());
          }
        }
      }
      return nullptr;
    };

    for (const auto &Idx : Selections)
      if (llvm::Value *P = EmitPtrFromIndices(Idx))
        TargetPtrs.push_back(P);

    if (TargetPtrs.empty())
      return nullptr;

    for (unsigned I = 0; I < E->getNumArgs(); ++I) {
      const Expr *Arg = E->getArg(I);
      if (!Arg->getType()->isPointerType())
        continue;
      bool IsTarget = false;
      for (const auto &Idx : Selections) {
        if (Idx.empty() || Idx[0] != I)
          continue;
        if (Idx.size() == 1 || (Idx.size() == 2 && Idx[1] == 0)) {
          IsTarget = true;
          break;
        }
      }
      if (!IsTarget)
        OtherPtrs.push_back(CGF.EmitScalarExpr(Arg));
    }

    auto CreatePtrArray = [&](llvm::ArrayRef<llvm::Value *> Ptrs,
                              const llvm::Twine &Name) -> llvm::Value * {
      if (Ptrs.empty())
        return llvm::ConstantPointerNull::get(
            llvm::PointerType::getUnqual(CGF.CGM.Int8PtrTy));
      llvm::ArrayType *ArrTy = llvm::ArrayType::get(CGF.CGM.Int8PtrTy, Ptrs.size());
      llvm::AllocaInst *ArrAlloca = CGF.Builder.CreateAlloca(ArrTy, nullptr, Name);
      llvm::Value *Zero = llvm::ConstantInt::get(CGF.IntTy, 0);
      for (unsigned Idx = 0; Idx < Ptrs.size(); ++Idx) {
        llvm::Value *CastPtr = CGF.Builder.CreateBitCast(Ptrs[Idx], CGF.CGM.Int8PtrTy);
        llvm::Value *ElemPtr = CGF.Builder.CreateInBoundsGEP(
            ArrTy, ArrAlloca, {Zero, llvm::ConstantInt::get(CGF.IntTy, Idx)});
        CGF.Builder.CreateDefaultAlignedStore(CastPtr, ElemPtr);
      }
      llvm::Value *FirstElemPtr = CGF.Builder.CreateInBoundsGEP(ArrTy, ArrAlloca,
                                                                {Zero, Zero});
      return CGF.Builder.CreateBitCast(
          FirstElemPtr, llvm::PointerType::getUnqual(CGF.CGM.Int8PtrTy));
    };

    llvm::Value *TargetsPtr = CreatePtrArray(TargetPtrs, "targets");
    llvm::Value *OthersPtr = CreatePtrArray(OtherPtrs, "others");
    llvm::Type *IntTy = CGF.IntTy;
    llvm::Type *Int8PtrPtrTy = llvm::PointerType::getUnqual(CGF.CGM.Int8PtrTy);
    llvm::FunctionType *CheckPtrTy = llvm::FunctionType::get(
        CGF.Builder.getInt1Ty(), {IntTy, Int8PtrPtrTy, IntTy, Int8PtrPtrTy}, false);
    llvm::FunctionCallee CheckPtrFn =
        CGF.CGM.CreateRuntimeFunction(CheckPtrTy, "check_ptr_sets");
    llvm::Value *NumTargets = llvm::ConstantInt::get(IntTy, TargetPtrs.size());
    llvm::Value *NumOthers = llvm::ConstantInt::get(IntTy, OtherPtrs.size());
    llvm::CallBase *AliasCall = CGF.EmitRuntimeCallOrInvoke(
        CheckPtrFn, {NumTargets, TargetsPtr, NumOthers, OthersPtr});
    return CGF.Builder.CreateNot(AliasCall, "noalias");
  };

  llvm::Value *ConstOK = ConstProfile ? EmitConstCheck(*ConstProfile) : nullptr;
  llvm::Value *NoaliasOK = NoaliasProfile ? EmitNoaliasCheck(*NoaliasProfile) : nullptr;

  if (ConstOK && NoaliasOK) {
    llvm::BasicBlock *ConstBlock = CGF.createBasicBlock("kcall.const_ok");
    llvm::BasicBlock *NoConstBlock = CGF.createBasicBlock("kcall.const_fail");
    llvm::BasicBlock *CombinedBlock = CGF.createBasicBlock("kcall.const_noalias");
    llvm::BasicBlock *ConstOnlyBlock = CGF.createBasicBlock("kcall.const_only");
    llvm::BasicBlock *NoaliasOnlyBlock = CGF.createBasicBlock("kcall.noalias_only");
    llvm::BasicBlock *OriginalBlock = CGF.createBasicBlock("kcall.original");

    CGF.Builder.CreateCondBr(ConstOK, ConstBlock, NoConstBlock);

    CGF.EmitBlock(ConstBlock);
    CGF.Builder.CreateCondBr(NoaliasOK, CombinedBlock, ConstOnlyBlock);

    CGF.EmitBlock(NoConstBlock);
    CGF.Builder.CreateCondBr(NoaliasOK, NoaliasOnlyBlock, OriginalBlock);

    CGF.EmitBlock(CombinedBlock);
    EmitSelectedCall("_const_noalias");
    CGF.EmitBranch(ContBlock);

    CGF.EmitBlock(ConstOnlyBlock);
    EmitSelectedCall("_const");
    CGF.EmitBranch(ContBlock);

    CGF.EmitBlock(NoaliasOnlyBlock);
    EmitSelectedCall("_noalias");
    CGF.EmitBranch(ContBlock);

    CGF.EmitBlock(OriginalBlock);
    EmitOriginalCall();
    CGF.EmitBranch(ContBlock);
  } else if (ConstOK) {
    llvm::BasicBlock *UseConstBlock = CGF.createBasicBlock("kcall.const");
    llvm::BasicBlock *UseOriginalBlock = CGF.createBasicBlock("kcall.original");
    CGF.Builder.CreateCondBr(ConstOK, UseConstBlock, UseOriginalBlock);

    CGF.EmitBlock(UseConstBlock);
    EmitSelectedCall("_const");
    CGF.EmitBranch(ContBlock);

    CGF.EmitBlock(UseOriginalBlock);
    EmitOriginalCall();
    CGF.EmitBranch(ContBlock);
  } else if (NoaliasOK) {
    llvm::BasicBlock *UseNoaliasBlock = CGF.createBasicBlock("kcall.noalias");
    llvm::BasicBlock *UseOriginalBlock = CGF.createBasicBlock("kcall.original");
    CGF.Builder.CreateCondBr(NoaliasOK, UseNoaliasBlock, UseOriginalBlock);

    CGF.EmitBlock(UseNoaliasBlock);
    EmitSelectedCall("_noalias");
    CGF.EmitBranch(ContBlock);

    CGF.EmitBlock(UseOriginalBlock);
    EmitOriginalCall();
    CGF.EmitBranch(ContBlock);
  } else {
    EmitOriginalCall();
    CGF.EmitBranch(ContBlock);
  }

  CGF.EmitBlock(ContBlock);
  Eval.end(CGF);
  return RValue::get(nullptr);
}

