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

static llvm::StringMap<KernelProfileInfo> KernelProfileMap;
static bool KernelProfileLoaded = false;

static void loadCudaKernelProfile(const CodeGenModule &CGM) {
  if (KernelProfileLoaded)
    return;
  KernelProfileLoaded = true;

  std::string ProfilePath;
  // 1) Environment variable takes priority.
  if (const char *Env = std::getenv("CUDA_KERNEL_PROFILE"))
    ProfilePath = Env;

  // 2) Fallback: scan backend options for "-cuda-kernel-profile=".
  if (ProfilePath.empty()) {
    for (const std::string &Opt : CGM.getCodeGenOpts().CommandLineArgs) {
      llvm::StringRef S(Opt);
      if (S.consume_front("-cuda-kernel-profile=")) {
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
  
  if (FD && CGF.CGM.getCodeGenOpts().CudaKernelNoalias) {
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
    if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E->getCallee())) {
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
  } else {
    // No conditional logic needed, use original call
    CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
  }
  
  CGF.EmitBranch(ContBlock);

  CGF.EmitBlock(ContBlock);
  eval.end(CGF);

  return RValue::get(nullptr);
}
