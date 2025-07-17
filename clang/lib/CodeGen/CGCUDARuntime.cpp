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

using namespace clang;
using namespace CodeGen;

//===----------------------------------------------------------------------===//
// Helper utilities for CUDA kernel profile parsing (shared with LLVM pass)
//===----------------------------------------------------------------------===//

namespace {
struct KernelProfileInfo {
  std::vector<unsigned> TargetPointerIndices; ///< Pointer parameter indices
};

struct ScalarConstInfo {
  unsigned index;     // Argument index (0-based)
  std::string value;  // Common constant value as string
  double ratio;       // Frequency ratio (0.0 to 1.0)
};

struct KernelConstProfileInfo {
  std::vector<ScalarConstInfo> commonScalars;
};

static llvm::StringMap<KernelProfileInfo> KernelProfileMap;
static llvm::StringMap<KernelConstProfileInfo> KernelConstProfileMap;
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

  if (llvm::json::Array *HotKernels = RootObj->getArray("hot_kernels")) {
    for (llvm::json::Value &HKVal : *HotKernels) {
      llvm::json::Object *HKObj = HKVal.getAsObject();
      if (!HKObj)
        continue;
      auto NameVal = HKObj->getString("name");
      if (!NameVal)
        continue;

      KernelProfileInfo Info;
      if (llvm::json::Array *NoaliasArr = HKObj->getArray("noalias_pointers")) {
        for (llvm::json::Value &PtrVal : *NoaliasArr) {
          llvm::json::Object *PtrObj = PtrVal.getAsObject();
          if (!PtrObj)
            continue;
          auto ArgVal = PtrObj->get("arg");
          if (!ArgVal)
            continue;
          if (auto ArgInt = ArgVal->getAsInteger())
            Info.TargetPointerIndices.push_back(static_cast<unsigned>(*ArgInt));
        }
      }
      if (!Info.TargetPointerIndices.empty())
        KernelProfileMap[*NameVal] = std::move(Info);
      
      // Parse common_scalars for constant propagation
      KernelConstProfileInfo ConstInfo;
      if (llvm::json::Array *ScalarsArr = HKObj->getArray("common_scalars")) {
        for (llvm::json::Value &ScalarVal : *ScalarsArr) {
          llvm::json::Object *ScalarObj = ScalarVal.getAsObject();
          if (!ScalarObj)
            continue;
          
          ScalarConstInfo SCI;
          if (auto ArgVal = ScalarObj->get("arg")) {
            if (auto ArgInt = ArgVal->getAsInteger()) {
              SCI.index = static_cast<unsigned>(*ArgInt);
            } else {
              continue;
            }
          } else {
            continue;
          }
          
          if (auto ValueVal = ScalarObj->getString("value")) {
            SCI.value = *ValueVal;
          } else {
            continue;
          }
          
          if (auto RatioVal = ScalarObj->get("ratio")) {
            if (auto RatioNum = RatioVal->getAsNumber()) {
              SCI.ratio = *RatioNum;
            } else {
              SCI.ratio = 1.0;
            }
          } else {
            SCI.ratio = 1.0;
          }
          
          ConstInfo.commonScalars.push_back(SCI);
        }
      }
      
      if (!ConstInfo.commonScalars.empty())
        KernelConstProfileMap[*NameVal] = std::move(ConstInfo);
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

         // Retrieve mangled kernel name to match profile
     std::string MangledName = CGF.CGM.getMangledName(GlobalDecl(FD)).str();
     auto ConstProfileIt = KernelConstProfileMap.find(MangledName);

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
         
         // Fill expected values
         for (size_t i = 0; i < Profile.commonScalars.size(); ++i) {
           int64_t ExpectedValue = std::stoll(Profile.commonScalars[i].value);
           llvm::Value *ExpectedVal = llvm::ConstantInt::get(CGF.Builder.getInt64Ty(), ExpectedValue);
           llvm::Value *ExpectedPtr = CGF.Builder.CreateInBoundsGEP(
               Int64ArrayTy, ExpectedArray.emitRawPointer(CGF), {CGF.Builder.getInt32(0), CGF.Builder.getInt32(i)});
           Address ExpectedAddr = Address(ExpectedPtr, CGF.Builder.getInt64Ty(), CGF.getPointerAlign());
           CGF.Builder.CreateStore(ExpectedVal, ExpectedAddr);
         }
         
         // Fill actual values from function arguments
         for (size_t i = 0; i < Profile.commonScalars.size(); ++i) {
           unsigned ArgIndex = Profile.commonScalars[i].index;
           if (ArgIndex < E->getNumArgs()) {
             llvm::Value *ActualVal = CGF.EmitAnyExpr(E->getArg(ArgIndex)).getScalarVal();
             // Convert to int64
             if (ActualVal->getType()->isIntegerTy()) {
               ActualVal = CGF.Builder.CreateSExtOrTrunc(ActualVal, CGF.Builder.getInt64Ty());
             } else if (ActualVal->getType()->isFloatingPointTy()) {
               ActualVal = CGF.Builder.CreateFPToSI(ActualVal, CGF.Builder.getInt64Ty());
             }
             llvm::Value *ActualPtr = CGF.Builder.CreateInBoundsGEP(
                 Int64ArrayTy, ActualArray.emitRawPointer(CGF), {CGF.Builder.getInt32(0), CGF.Builder.getInt32(i)});
             Address ActualAddr = Address(ActualPtr, CGF.Builder.getInt64Ty(), CGF.getPointerAlign());
             CGF.Builder.CreateStore(ActualVal, ActualAddr);
           }
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
    auto ProfileIt = KernelProfileMap.find(MangledName);

    if (ProfileIt == KernelProfileMap.end()) {
      // Try without the __device_stub__ prefix which Clang adds to host stubs.
      llvm::StringRef NameRef(MangledName);
      size_t StubPos = NameRef.find("__device_stub__");
      if (StubPos != llvm::StringRef::npos) {
        llvm::StringRef Suffix = NameRef.substr(StubPos + strlen("__device_stub__"));
        // Try to find any kernel whose mangled name ends with the suffix.
        for (auto It = KernelProfileMap.begin(); It != KernelProfileMap.end(); ++It) {
          if (llvm::StringRef(It->getKey()).ends_with(Suffix)) {
            ProfileIt = It;
            break;
          }
        }
      }
    }

    // If no profile information or no target pointers recorded, fall back.
    if (ProfileIt == KernelProfileMap.end() ||
        ProfileIt->second.TargetPointerIndices.empty()) {
      CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
      CGF.EmitBranch(ContBlock);
      CGF.EmitBlock(ContBlock);
      eval.end(CGF);
      return RValue::get(nullptr);
    }

    const std::vector<unsigned> &TargetIndices =
        ProfileIt->second.TargetPointerIndices;

    llvm::BasicBlock *useNoaliasBlock = CGF.createBasicBlock("use_noalias");
    llvm::BasicBlock *useOriginalBlock = CGF.createBasicBlock("use_original");
    llvm::BasicBlock *afterKernelCallBlock =
        CGF.createBasicBlock("after_kernel_call");

    // Separate pointer arguments into target set and other set.
    llvm::SmallVector<llvm::Value *, 8> TargetPtrs;
    llvm::SmallVector<llvm::Value *, 8> OtherPtrs;

    for (unsigned i = 0; i < E->getNumArgs(); ++i) {
      const Expr *Arg = E->getArg(i);
      if (!Arg->getType()->isPointerType())
        continue;
      llvm::Value *ArgVal = CGF.EmitScalarExpr(Arg);
      if (llvm::is_contained(TargetIndices, i))
        TargetPtrs.push_back(ArgVal);
      else
        OtherPtrs.push_back(ArgVal);
    }

    // Generate runtime alias check whenever there is at least one target pointer.
    llvm::Value *noAliasCondition = nullptr;
    if (!TargetPtrs.empty()) {
      auto createPtrArray = [&](llvm::ArrayRef<llvm::Value *> Ptrs,
                                const llvm::Twine &Name) -> llvm::Value * {
        if (Ptrs.empty())
          return llvm::ConstantPointerNull::get(
              llvm::PointerType::getUnqual(CGF.CGM.Int8PtrTy));

        llvm::ArrayType *ArrTy =
            llvm::ArrayType::get(CGF.CGM.Int8PtrTy, Ptrs.size());
        llvm::AllocaInst *ArrAlloca =
            CGF.Builder.CreateAlloca(ArrTy, nullptr, Name);
        llvm::Value *Zero = llvm::ConstantInt::get(CGF.IntTy, 0);
        for (unsigned idx = 0; idx < Ptrs.size(); ++idx) {
          llvm::Value *CastPtr =
              CGF.Builder.CreateBitCast(Ptrs[idx], CGF.CGM.Int8PtrTy);
          llvm::Value *ElemPtr = CGF.Builder.CreateInBoundsGEP(
              ArrTy, ArrAlloca,
              {Zero, llvm::ConstantInt::get(CGF.IntTy, idx)});
          CGF.Builder.CreateDefaultAlignedStore(CastPtr, ElemPtr);
        }
        llvm::Value *FirstElemPtr = CGF.Builder.CreateInBoundsGEP(
            ArrTy, ArrAlloca, {Zero, Zero});
        llvm::Type *Int8PtrPtrTy =
            llvm::PointerType::getUnqual(CGF.CGM.Int8PtrTy);
        return CGF.Builder.CreateBitCast(FirstElemPtr, Int8PtrPtrTy);
      };

      llvm::Value *TargetsPtr = createPtrArray(TargetPtrs, "targets");
      llvm::Value *OthersPtr = createPtrArray(OtherPtrs, "others");

      llvm::Type *IntTy = CGF.IntTy; // i32
      llvm::Type *Int8PtrPtrTy = llvm::PointerType::getUnqual(CGF.CGM.Int8PtrTy);

      // bool check_ptr_sets(int, i8**, int, i8**)
      llvm::FunctionType *CheckPtrTy = llvm::FunctionType::get(
          CGF.Builder.getInt1Ty(), {IntTy, Int8PtrPtrTy, IntTy, Int8PtrPtrTy},
          /*isVarArg=*/false);
      llvm::FunctionCallee CheckPtrFn =
          CGF.CGM.CreateRuntimeFunction(CheckPtrTy, "check_ptr_sets");

      llvm::Value *NumTargets =
          llvm::ConstantInt::get(IntTy, TargetPtrs.size());
      llvm::Value *NumOthers = llvm::ConstantInt::get(IntTy, OtherPtrs.size());

      llvm::CallBase *AliasCall = CGF.EmitRuntimeCallOrInvoke(
          CheckPtrFn, {NumTargets, TargetsPtr, NumOthers, OthersPtr});

      // AliasCall == true  => aliasing exists  => use ORIGINAL kernel
      // AliasCall == false => no aliasing      => use NOALIAS kernel
      noAliasCondition = CGF.Builder.CreateNot(AliasCall, "noalias");
    } else {
      // No target pointer recorded – fall back conservatively.
      noAliasCondition = llvm::ConstantInt::getFalse(CGF.Builder.getContext());
    }

    CGF.Builder.CreateCondBr(noAliasCondition, useNoaliasBlock,
                             useOriginalBlock);

    //===------------------------------------------------------------------===//
    // Noalias branch – call the specialised stub with suffix "_noalias".
    //===------------------------------------------------------------------===//
    CGF.EmitBlock(useNoaliasBlock);
    const Expr *Callee = E->getCallee();
    // Strip away implicit casts to get to the underlying DeclRefExpr
    const Expr *UnderlyingCallee = Callee->IgnoreParenImpCasts();
    
    if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(UnderlyingCallee)) {
      if (const FunctionDecl *CalledFD = dyn_cast<FunctionDecl>(DRE->getDecl())) {
        std::string NoaliasStubName =
            CGF.CGM.getMangledName(GlobalDecl(CalledFD)).str() + "_noalias";
        if (llvm::Function *NoaliasStub =
                CGF.CGM.getModule().getFunction(NoaliasStubName)) {
          CallArgList Args;
          for (const Expr *Arg : E->arguments())
            Args.add(CGF.EmitAnyExpr(Arg), Arg->getType());

          const CGFunctionInfo &FnInfo = CGF.CGM.getTypes().arrangeFreeFunctionCall(
              Args, CalledFD->getType()->castAs<FunctionType>(), /*ChainCall=*/false);
          CGF.EmitCall(FnInfo, CGCallee::forDirect(NoaliasStub), ReturnValueSlot(),
                       Args);
        } else {
          // Fallback – noalias stub missing.
          CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
        }
      } else {
        CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
      }
    } else {
      CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
    }
    CGF.EmitBranch(afterKernelCallBlock);

    //===------------------------------------------------------------------===//
    // Original branch
    //===------------------------------------------------------------------===//
    CGF.EmitBlock(useOriginalBlock);
    CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
    CGF.EmitBranch(afterKernelCallBlock);

    CGF.EmitBlock(afterKernelCallBlock);
  } else if (FD && CGF.CGM.getCodeGenOpts().CudaKernelConst) {
    // Check if we need to generate conditional logic for const propagation
    loadCudaKernelProfile(CGF.CGM);

    // Retrieve mangled kernel name to match profile
    std::string MangledName = CGF.CGM.getMangledName(GlobalDecl(FD)).str();
    auto ConstProfileIt = KernelConstProfileMap.find(MangledName);

    if (ConstProfileIt == KernelConstProfileMap.end()) {
      // Try without the __device_stub__ prefix
      llvm::StringRef NameRef(MangledName);
      size_t StubPos = NameRef.find("__device_stub__");
      if (StubPos != llvm::StringRef::npos) {
        llvm::StringRef Suffix = NameRef.substr(StubPos + strlen("__device_stub__"));
        for (auto It = KernelConstProfileMap.begin(); It != KernelConstProfileMap.end(); ++It) {
          if (llvm::StringRef(It->getKey()).ends_with(Suffix)) {
            ConstProfileIt = It;
            break;
          }
        }
      }
    }

    // If no profile information or no common scalars, fall back to original
    if (ConstProfileIt == KernelConstProfileMap.end() ||
        ConstProfileIt->second.commonScalars.empty()) {
      CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
    } else {
      const std::vector<ScalarConstInfo> &CommonScalars = ConstProfileIt->second.commonScalars;

      llvm::BasicBlock *useConstBlock = CGF.createBasicBlock("use_const");
      llvm::BasicBlock *useOriginalBlock = CGF.createBasicBlock("use_original");
      llvm::BasicBlock *afterConstCallBlock = CGF.createBasicBlock("after_const_call");

      // Create runtime check for constant values
      llvm::Value *constCondition = nullptr;
      if (!CommonScalars.empty()) {
        // Create arrays for expected values and actual values
        llvm::SmallVector<llvm::Value *, 8> ExpectedValues;
        llvm::SmallVector<llvm::Value *, 8> ActualValues;

        for (const ScalarConstInfo &SCI : CommonScalars) {
          if (SCI.index < E->getNumArgs()) {
            const Expr *Arg = E->getArg(SCI.index);
            llvm::Value *ActualVal = CGF.EmitScalarExpr(Arg);
            ActualValues.push_back(ActualVal);
            
            // Create expected constant value
            llvm::Value *ExpectedVal = nullptr;
            if (Arg->getType()->isIntegerType()) {
              int64_t val = std::stoll(SCI.value);
              ExpectedVal = llvm::ConstantInt::get(ActualVal->getType(), val);
            } else if (Arg->getType()->isFloatingType()) {
              double val = std::stod(SCI.value);
              ExpectedVal = llvm::ConstantFP::get(ActualVal->getType(), val);
            }
            
            if (ExpectedVal) {
              ExpectedValues.push_back(ExpectedVal);
            }
          }
        }

        if (!ExpectedValues.empty()) {
          auto createValueArray = [&](llvm::ArrayRef<llvm::Value *> Values,
                                     const llvm::Twine &Name) -> llvm::Value * {
            llvm::ArrayType *ArrTy = llvm::ArrayType::get(CGF.CGM.Int64Ty, Values.size());
            llvm::AllocaInst *ArrAlloca = CGF.Builder.CreateAlloca(ArrTy, nullptr, Name);
            llvm::Value *Zero = llvm::ConstantInt::get(CGF.IntTy, 0);
            for (unsigned idx = 0; idx < Values.size(); ++idx) {
              llvm::Value *CastVal = CGF.Builder.CreateIntCast(Values[idx], CGF.CGM.Int64Ty, true);
              llvm::Value *ElemPtr = CGF.Builder.CreateInBoundsGEP(
                  ArrTy, ArrAlloca, {Zero, llvm::ConstantInt::get(CGF.IntTy, idx)});
              CGF.Builder.CreateDefaultAlignedStore(CastVal, ElemPtr);
            }
            llvm::Value *FirstElemPtr = CGF.Builder.CreateInBoundsGEP(
                ArrTy, ArrAlloca, {Zero, Zero});
            return CGF.Builder.CreateBitCast(FirstElemPtr, 
                                            llvm::PointerType::getUnqual(CGF.CGM.Int64Ty));
          };

          llvm::Value *ExpectedPtr = createValueArray(ExpectedValues, "expected");
          llvm::Value *ActualPtr = createValueArray(ActualValues, "actual");

          llvm::Type *IntTy = CGF.IntTy;
          llvm::Type *Int64PtrTy = llvm::PointerType::getUnqual(CGF.CGM.Int64Ty);

          // bool check_const(int, i64*, i64*)
          llvm::FunctionType *CheckConstTy = llvm::FunctionType::get(
              CGF.Builder.getInt1Ty(), {IntTy, Int64PtrTy, Int64PtrTy}, false);
          llvm::FunctionCallee CheckConstFn =
              CGF.CGM.CreateRuntimeFunction(CheckConstTy, "check_const");

          llvm::Value *NumValues = llvm::ConstantInt::get(IntTy, ExpectedValues.size());
          llvm::CallBase *ConstCall = CGF.EmitRuntimeCallOrInvoke(
              CheckConstFn, {NumValues, ExpectedPtr, ActualPtr});

          // ConstCall == true  => values match     => use CONST kernel
          // ConstCall == false => values mismatch  => use ORIGINAL kernel
          constCondition = ConstCall;
        } else {
          constCondition = llvm::ConstantInt::getFalse(CGF.Builder.getContext());
        }
      } else {
        constCondition = llvm::ConstantInt::getFalse(CGF.Builder.getContext());
      }

      CGF.Builder.CreateCondBr(constCondition, useConstBlock, useOriginalBlock);

      //===------------------------------------------------------------------===//
      // Const branch – call the specialised stub with suffix "_const".
      //===------------------------------------------------------------------===//
      CGF.EmitBlock(useConstBlock);
      const Expr *Callee = E->getCallee();
      // Strip away implicit casts to get to the underlying DeclRefExpr
      const Expr *UnderlyingCallee = Callee->IgnoreParenImpCasts();
      
      if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(UnderlyingCallee)) {
        if (const FunctionDecl *CalledFD = dyn_cast<FunctionDecl>(DRE->getDecl())) {
          std::string ConstStubName =
              CGF.CGM.getMangledName(GlobalDecl(CalledFD)).str() + "_const";
          if (llvm::Function *ConstStub =
                  CGF.CGM.getModule().getFunction(ConstStubName)) {
            CallArgList Args;
            for (const Expr *Arg : E->arguments())
              Args.add(CGF.EmitAnyExpr(Arg), Arg->getType());

            const CGFunctionInfo &FnInfo = CGF.CGM.getTypes().arrangeFreeFunctionCall(
                Args, CalledFD->getType()->castAs<FunctionType>(), /*ChainCall=*/false);
            CGF.EmitCall(FnInfo, CGCallee::forDirect(ConstStub), ReturnValueSlot(), Args);
          } else {
            // Fallback – const stub missing.
            CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
          }
        } else {
          CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
        }
      } else {
        CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
      }
      CGF.EmitBranch(afterConstCallBlock);

      //===------------------------------------------------------------------===//
      // Original branch
      //===------------------------------------------------------------------===//
      CGF.EmitBlock(useOriginalBlock);
      CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
      CGF.EmitBranch(afterConstCallBlock);

      CGF.EmitBlock(afterConstCallBlock);
    }
  } else {
    // No conditional logic needed, use original call
    CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
  }
  
  CGF.EmitBranch(ContBlock);

  CGF.EmitBlock(ContBlock);
  eval.end(CGF);

  return RValue::get(nullptr);
}
