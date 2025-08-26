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
};

struct KernelConstProfileInfo {
  std::vector<ScalarConstInfo> commonScalars;
};

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
    auto addConstItem = [&](llvm::StringRef Kernel, std::vector<unsigned> Indices, std::string ValueStr) {
      KernelConstProfileInfo &Slot = KernelConstProfileMap[Kernel];
      ScalarConstInfo Item;
      Item.index = Indices.empty() ? 0u : Indices.front();
      Item.value = std::move(ValueStr);
      Item.ratio = 1.0; // final selection – treat as always-on
      Item.indices = std::move(Indices);
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
              addConstItem(KernelName, std::move(IndicesVec), std::move(ValueStr));
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
                addConstItem(KName, std::move(IndicesVec), std::move(ValueStr));
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

  CodeGenFunction::ConditionalEvaluation eval(CGF);
  CGF.EmitBranchOnBoolExpr(E->getConfig(), ContBlock, ConfigOKBlock,
                           /*TrueCount=*/0);

  eval.begin(CGF);
  CGF.EmitBlock(ConfigOKBlock);
  
  // Check if we need to generate conditional logic for noalias selection
  const FunctionDecl *FD = dyn_cast<FunctionDecl>(E->getCalleeDecl());
  

  
     if (FD && CGF.CGM.getCodeGenOpts().CudaKernelConst) {
     // Check if we need to generate conditional logic for const propagation
     loadCudaKernelProfile(CGF.CGM);

         // Retrieve names to match profile (mangled and simple)
     std::string MangledName = CGF.CGM.getMangledName(GlobalDecl(FD)).str();
     std::string SimpleName = FD->getNameAsString();
     auto ConstProfileIt = KernelConstProfileMap.find(MangledName);
     if (ConstProfileIt == KernelConstProfileMap.end())
       ConstProfileIt = KernelConstProfileMap.find(SimpleName);

      if (ConstProfileIt == KernelConstProfileMap.end()) {
        // Try without the __device_stub__ prefix which Clang adds to host stubs.
        llvm::StringRef NameRef(MangledName);
        size_t StubPos = NameRef.find("__device_stub__");
        if (StubPos != llvm::StringRef::npos) {
          llvm::StringRef Suffix = NameRef.substr(StubPos + strlen("__device_stub__"));
          // Try to find any kernel whose mangled name ends with the suffix.
          for (auto It = KernelConstProfileMap.begin(); It != KernelConstProfileMap.end(); ++It) {
            if (llvm::StringRef(It->getKey()).ends_with(Suffix)) {
              ConstProfileIt = It;
              break;
            }
          }
        }
       // Also try matching by simple name suffix
       if (ConstProfileIt == KernelConstProfileMap.end()) {
         for (auto It = KernelConstProfileMap.begin(); It != KernelConstProfileMap.end(); ++It) {
           if (llvm::StringRef(It->getKey()) == SimpleName ||
               llvm::StringRef(It->getKey()).ends_with(SimpleName)) {
             ConstProfileIt = It;
             break;
           }
         }
       }
      }

         if (ConstProfileIt != KernelConstProfileMap.end()) {
       const KernelConstProfileInfo &Profile = ConstProfileIt->second;
       
       if (!Profile.commonScalars.empty()) {
         // Generate conditional logic for const propagation
         llvm::BasicBlock *useConstBlock = CGF.createBasicBlock("use_const");
         llvm::BasicBlock *useOriginalBlock = CGF.createBasicBlock("use_original");
         llvm::BasicBlock *afterConstCallBlock = CGF.createBasicBlock("after_const_call");

         // Create check_const function call
         llvm::Type *BoolTy = CGF.Builder.getInt1Ty();
         llvm::Type *IntTy = CGF.IntTy;
         llvm::Type *Int64PtrTy = llvm::PointerType::getUnqual(CGF.Builder.getInt64Ty());
         
         llvm::FunctionType *CheckConstTy = llvm::FunctionType::get(
             BoolTy, {IntTy, Int64PtrTy, Int64PtrTy}, false);
         llvm::FunctionCallee CheckConstFn =
             CGF.CGM.CreateRuntimeFunction(CheckConstTy, "check_const");

         // Prepare expected and actual values
         llvm::Constant *NumValues = llvm::ConstantInt::get(IntTy, Profile.commonScalars.size());
         
         // Create arrays for expected and actual values
         llvm::ArrayType *Int64ArrayTy = llvm::ArrayType::get(CGF.Builder.getInt64Ty(), Profile.commonScalars.size());
         Address ExpectedArray = CGF.CreateTempAlloca(Int64ArrayTy, CGF.getPointerAlign(), "expected_values");
         Address ActualArray = CGF.CreateTempAlloca(Int64ArrayTy, CGF.getPointerAlign(), "actual_values");
         
         // Fill expected values (support integer or floating literal strings)
         for (size_t i = 0; i < Profile.commonScalars.size(); ++i) {
           const std::string &S = Profile.commonScalars[i].value;
           int64_t ExpectedValue = 0;
           if (S.find_first_of(".eE") != std::string::npos) {
             double dv = std::stod(S);
             ExpectedValue = static_cast<int64_t>(dv);
           } else {
             ExpectedValue = std::stoll(S);
           }
           llvm::Value *ExpectedVal = llvm::ConstantInt::get(CGF.Builder.getInt64Ty(), ExpectedValue);
           llvm::Value *ExpectedPtr = CGF.Builder.CreateInBoundsGEP(
               Int64ArrayTy, ExpectedArray.emitRawPointer(CGF), {CGF.Builder.getInt32(0), CGF.Builder.getInt32(i)});
           Address ExpectedAddr = Address(ExpectedPtr, CGF.Builder.getInt64Ty(), CGF.getPointerAlign());
           CGF.Builder.CreateStore(ExpectedVal, ExpectedAddr);
         }
         
         // Fill actual values from function arguments
         for (size_t i = 0; i < Profile.commonScalars.size(); ++i) {
           const ScalarConstInfo &SCI = Profile.commonScalars[i];
           const std::vector<unsigned> &Path = SCI.indices;
           if (Path.empty()) continue;
 
           unsigned ArgIndex = Path[0];
           if (ArgIndex >= E->getNumArgs()) continue;
           const Expr *ArgE = E->getArg(ArgIndex);
 
           llvm::Value *ActualVal = nullptr;
           if (Path.size() == 1 || (Path.size() == 2 && Path[1] == 0)) {
             // This is a top-level scalar argument.
             ActualVal = CGF.EmitAnyExpr(ArgE).getScalarVal();
           } else if (Path.size() == 2) {
             // This is a struct member, identified by byte offset.
             uint64_t Offset = Path[1];
             LValue BaseLV = CGF.EmitLValue(ArgE);
             llvm::Value *BasePtr = BaseLV.getPointer(CGF);
             llvm::Value *OffsetVal = llvm::ConstantInt::get(CGF.SizeTy, Offset);

             // GEP on the i8* pointer
             llvm::Value *GEPPtr = CGF.Builder.CreateInBoundsGEP(
                 CGF.Builder.getInt8Ty(), BasePtr, OffsetVal);

             // We need to know the type of the member to load.
             // This is tricky without the FieldDecl. We assume for now it's a 64-bit integer
             // as that's what the check function expects.
             llvm::Type *DestPtrTy = llvm::PointerType::getUnqual(CGF.Builder.getInt64Ty());
             llvm::Value *CastedPtr = CGF.Builder.CreateBitCast(GEPPtr, DestPtrTy);
             
             ActualVal = CGF.Builder.CreateLoad(Address(CastedPtr, CGF.Builder.getInt64Ty(), CGF.getPointerAlign()));
           }

           if (!ActualVal) continue;

           // Convert to int64 for the check function
           if (ActualVal->getType()->isIntegerTy()) {
             ActualVal = CGF.Builder.CreateSExtOrTrunc(ActualVal, CGF.Builder.getInt64Ty());
           } else if (ActualVal->getType()->isFloatingPointTy()) {
             ActualVal = CGF.Builder.CreateFPToSI(ActualVal, CGF.Builder.getInt64Ty());
           } else {
             continue;
           }
           llvm::Value *ActualPtr = CGF.Builder.CreateInBoundsGEP(
               Int64ArrayTy, ActualArray.emitRawPointer(CGF), {CGF.Builder.getInt32(0), CGF.Builder.getInt32(i)});
           Address ActualAddr = Address(ActualPtr, CGF.Builder.getInt64Ty(), CGF.getPointerAlign());
           CGF.Builder.CreateStore(ActualVal, ActualAddr);
         }
         
         // Get pointers to arrays
         llvm::Value *ExpectedPtr = CGF.Builder.CreateBitCast(ExpectedArray.emitRawPointer(CGF), Int64PtrTy);
         llvm::Value *ActualPtr = CGF.Builder.CreateBitCast(ActualArray.emitRawPointer(CGF), Int64PtrTy);
         
         // Call check_const function
         llvm::Value *ShouldUseConst = CGF.EmitRuntimeCallOrInvoke(
             CheckConstFn, {NumValues, ExpectedPtr, ActualPtr});
         
         // Branch based on result
         CGF.Builder.CreateCondBr(ShouldUseConst, useConstBlock, useOriginalBlock);
         
         // Const branch - call optimized kernel
         CGF.EmitBlock(useConstBlock);
         
         const Expr *Callee = E->getCallee();
         // Strip away implicit casts to get to the underlying DeclRefExpr
         const Expr *UnderlyingCallee = Callee->IgnoreParenImpCasts();
         
         if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(UnderlyingCallee)) {
           if (const FunctionDecl *CalledFD = dyn_cast<FunctionDecl>(DRE->getDecl())) {
             // Build stub function name: __device_stub__kernel_name_const
             std::string BaseStubName = CGF.CGM.getMangledName(GlobalDecl(CalledFD, KernelReferenceKind::Stub)).str();
             std::string ConstStubName = BaseStubName + "_const";
             
             // Try to get or create the const stub function
             llvm::Function *ConstStub = CGF.CGM.getModule().getFunction(ConstStubName);
             
             if (!ConstStub) {
               // Create function declaration for const stub with same signature as original
               llvm::Function *OriginalStub = CGF.CGM.getModule().getFunction(BaseStubName);
               
               if (OriginalStub) {
                 ConstStub = llvm::Function::Create(OriginalStub->getFunctionType(), 
                                                    llvm::GlobalValue::ExternalLinkage, 
                                                    ConstStubName, &CGF.CGM.getModule());
               }
             }
             
             if (ConstStub) {
               CallArgList Args;
               for (const Expr *Arg : E->arguments())
                 Args.add(CGF.EmitAnyExpr(Arg), Arg->getType());
               
               const CGFunctionInfo &FnInfo = CGF.CGM.getTypes().arrangeFreeFunctionCall(
                   Args, CalledFD->getType()->castAs<FunctionType>(), false);
               CGF.EmitCall(FnInfo, CGCallee::forDirect(ConstStub), ReturnValueSlot(), Args);
             } else {
               // Fallback to original kernel
               CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
             }
           } else {
             CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
           }
         } else {
           CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
         }
         CGF.EmitBranch(afterConstCallBlock);
         
         // Original branch
         CGF.EmitBlock(useOriginalBlock);
         CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
         CGF.EmitBranch(afterConstCallBlock);
         
         CGF.EmitBlock(afterConstCallBlock);
       } else {
         // No const optimization needed
         CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
       }
     } else {
       // No profile found, use original call
       CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
     }
  } else if (FD && CGF.CGM.getCodeGenOpts().CudaKernelNoalias) {
    // Ensure profile has been parsed (only once per translation unit)
    loadCudaKernelProfile(CGF.CGM);

    // Retrieve mangled kernel name to match profile (device stub shares name)
    std::string MangledName = CGF.CGM.getMangledName(GlobalDecl(FD)).str();
    auto NAIt = KernelNoaliasProfileMap.find(MangledName);
    if (NAIt == KernelNoaliasProfileMap.end()) {
      // Try without the __device_stub__ prefix
      llvm::StringRef NameRef(MangledName);
      size_t StubPos = NameRef.find("__device_stub__");
      if (StubPos != llvm::StringRef::npos) {
        llvm::StringRef Suffix = NameRef.substr(StubPos + strlen("__device_stub__"));
        for (auto It = KernelNoaliasProfileMap.begin(); It != KernelNoaliasProfileMap.end(); ++It) {
          if (llvm::StringRef(It->getKey()).ends_with(Suffix)) { NAIt = It; break; }
        }
      }
      // Also try simple name
      if (NAIt == KernelNoaliasProfileMap.end()) {
        std::string SimpleName = FD->getNameAsString();
        for (auto It = KernelNoaliasProfileMap.begin(); It != KernelNoaliasProfileMap.end(); ++It) {
          if (llvm::StringRef(It->getKey()) == SimpleName || llvm::StringRef(It->getKey()).ends_with(SimpleName)) {
            NAIt = It; break;
          }
        }
      }
    }

    if (NAIt == KernelNoaliasProfileMap.end() || NAIt->second.empty()) {
      CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
      CGF.EmitBranch(ContBlock);
      CGF.EmitBlock(ContBlock);
      eval.end(CGF);
      return RValue::get(nullptr);
    }

    // Build targets/others pointer arrays from callsite
    llvm::SmallVector<llvm::Value*, 8> TargetPtrs;
    llvm::SmallVector<llvm::Value*, 8> OtherPtrs;

    auto EmitPtrFromIndices = [&](const std::vector<unsigned> &IdxVec) -> llvm::Value* {
      if (IdxVec.empty()) return nullptr;
      unsigned ArgIndex = IdxVec[0];
      if (ArgIndex >= E->getNumArgs()) return nullptr;
      const Expr *ArgE = E->getArg(ArgIndex);

      if (IdxVec.size() == 1 || (IdxVec.size() == 2 && IdxVec[1] == 0)) {
        // Top-level pointer argument
        if (!ArgE->getType()->isPointerType()) return nullptr;
        return CGF.EmitScalarExpr(ArgE);
      }

      if (IdxVec.size() == 2) {
        // Pointer is a struct member, identified by byte offset.
        uint64_t Offset = IdxVec[1];
        LValue LV = CGF.EmitLValue(ArgE);
        if (const auto *RT = LV.getType()->getAs<RecordType>()) {
          const RecordDecl *RD = RT->getDecl();
          const ASTContext &ASTCtx = CGF.getContext();
          const ASTRecordLayout &Layout = ASTCtx.getASTRecordLayout(RD);

          const FieldDecl *FD = nullptr;
          unsigned FieldNo = 0;
          for (const FieldDecl *Field : RD->fields()) {
            if (Layout.getFieldOffset(FieldNo) / 8 == Offset) {
              FD = Field;
              break;
            }
            FieldNo++;
          }

          if (FD && FD->getType()->isPointerType()) {
            LValue MemberLV = CGF.EmitLValueForField(LV, FD);
            return CGF.EmitLoadOfScalar(MemberLV.getAddress(), /*Volatile=*/false,
                                      FD->getType(), E->getExprLoc());
          }
        }
      }
      return nullptr;
    };

    // targets from selected indices
    for (const auto &Idx : NAIt->second) {
      if (llvm::Value *P = EmitPtrFromIndices(Idx)) TargetPtrs.push_back(P);
    }
    // others: all pointer args not in targets
    for (unsigned i = 0; i < E->getNumArgs(); ++i) {
      const Expr *Arg = E->getArg(i); if (!Arg->getType()->isPointerType()) continue;
      bool IsTarget = false;
      for (const auto &Idx : NAIt->second) {
        if (Idx.empty()) continue;
        if (Idx[0] != i) continue;
        if (Idx.size() == 1 || (Idx.size() == 2 && Idx[1] == 0)) { IsTarget = true; break; }
      }
      if (!IsTarget) OtherPtrs.push_back(CGF.EmitScalarExpr(Arg));
    }

    auto createPtrArray = [&](llvm::ArrayRef<llvm::Value*> Ptrs, const llvm::Twine &Name) -> llvm::Value* {
      if (Ptrs.empty()) return llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(CGF.CGM.Int8PtrTy));
      llvm::ArrayType *ArrTy = llvm::ArrayType::get(CGF.CGM.Int8PtrTy, Ptrs.size());
      llvm::AllocaInst *ArrAlloca = CGF.Builder.CreateAlloca(ArrTy, nullptr, Name);
      llvm::Value *Zero = llvm::ConstantInt::get(CGF.IntTy, 0);
      for (unsigned idx = 0; idx < Ptrs.size(); ++idx) {
        llvm::Value *CastPtr = CGF.Builder.CreateBitCast(Ptrs[idx], CGF.CGM.Int8PtrTy);
        llvm::Value *ElemPtr = CGF.Builder.CreateInBoundsGEP(ArrTy, ArrAlloca, {Zero, llvm::ConstantInt::get(CGF.IntTy, idx)});
        CGF.Builder.CreateDefaultAlignedStore(CastPtr, ElemPtr);
      }
      llvm::Value *FirstElemPtr = CGF.Builder.CreateInBoundsGEP(ArrTy, ArrAlloca, {Zero, Zero});
      llvm::Type *Int8PtrPtrTy = llvm::PointerType::getUnqual(CGF.CGM.Int8PtrTy);
      return CGF.Builder.CreateBitCast(FirstElemPtr, Int8PtrPtrTy);
    };

    llvm::Value *TargetsPtr = createPtrArray(TargetPtrs, "targets");
    llvm::Value *OthersPtr  = createPtrArray(OtherPtrs,  "others");

    llvm::Type *IntTy = CGF.IntTy; // i32
    llvm::Type *Int8PtrPtrTy = llvm::PointerType::getUnqual(CGF.CGM.Int8PtrTy);
    // bool check_ptr_sets(int, i8**, int, i8**)
    llvm::FunctionType *CheckPtrTy = llvm::FunctionType::get(
        CGF.Builder.getInt1Ty(), {IntTy, Int8PtrPtrTy, IntTy, Int8PtrPtrTy}, false);
    llvm::FunctionCallee CheckPtrFn = CGF.CGM.CreateRuntimeFunction(CheckPtrTy, "check_ptr_sets");

    llvm::Value *NumTargets = llvm::ConstantInt::get(IntTy, TargetPtrs.size());
    llvm::Value *NumOthers  = llvm::ConstantInt::get(IntTy, OtherPtrs.size());
    llvm::CallBase *AliasCall = CGF.EmitRuntimeCallOrInvoke(CheckPtrFn, {NumTargets, TargetsPtr, NumOthers, OthersPtr});
    llvm::Value *NoAlias = CGF.Builder.CreateNot(AliasCall, "noalias");

    llvm::BasicBlock *useNoaliasBlock = CGF.createBasicBlock("use_noalias");
    llvm::BasicBlock *useOriginalBlock = CGF.createBasicBlock("use_original");
    llvm::BasicBlock *afterKernelCallBlock = CGF.createBasicBlock("after_kernel_call");
    CGF.Builder.CreateCondBr(NoAlias, useNoaliasBlock, useOriginalBlock);

    // _noalias branch
    CGF.EmitBlock(useNoaliasBlock);
    {
      const Expr *Callee = E->getCallee();
      const Expr *UnderlyingCallee = Callee->IgnoreParenImpCasts();
      if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(UnderlyingCallee)) {
        if (const FunctionDecl *CalledFD = dyn_cast<FunctionDecl>(DRE->getDecl())) {
          std::string NoaliasStubName = CGF.CGM.getMangledName(GlobalDecl(CalledFD, KernelReferenceKind::Stub)).str() + "_noalias";
          if (llvm::Function *NoaliasStub = CGF.CGM.getModule().getFunction(NoaliasStubName)) {
            CallArgList Args; for (const Expr *Arg : E->arguments()) Args.add(CGF.EmitAnyExpr(Arg), Arg->getType());
            const CGFunctionInfo &FnInfo = CGF.CGM.getTypes().arrangeFreeFunctionCall(
                Args, CalledFD->getType()->castAs<FunctionType>(), /*ChainCall=*/false);
            CGF.EmitCall(FnInfo, CGCallee::forDirect(NoaliasStub), ReturnValueSlot(), Args);
          } else {
            CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
          }
        } else {
          CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
        }
      } else {
        CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
      }
    }
    CGF.EmitBranch(afterKernelCallBlock);

    // original branch
    CGF.EmitBlock(useOriginalBlock);
    CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
    CGF.EmitBranch(afterKernelCallBlock);

    CGF.EmitBlock(afterKernelCallBlock);

    CGF.EmitBranch(ContBlock);
    CGF.EmitBlock(ContBlock);
    eval.end(CGF);
    return RValue::get(nullptr);
  } else {
    // No conditional logic needed, use original call
    CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
  }
  
  CGF.EmitBranch(ContBlock);

  CGF.EmitBlock(ContBlock);
  eval.end(CGF);

  return RValue::get(nullptr);
}

