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

using namespace clang;
using namespace CodeGen;

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
    // Always generate runtime check when noalias is enabled
    llvm::BasicBlock *useNoaliasBlock = CGF.createBasicBlock("use_noalias");
    llvm::BasicBlock *useOriginalBlock = CGF.createBasicBlock("use_original");
    llvm::BasicBlock *afterKernelCallBlock = CGF.createBasicBlock("after_kernel_call");
    
    // Collect pointer arguments
    llvm::SmallVector<llvm::Value *, 4> ptrArgs;
    for (unsigned i = 0; i < E->getNumArgs(); ++i) {
      const Expr *Arg = E->getArg(i);
      if (Arg->getType()->isPointerType()) {
        llvm::Value *argValue = CGF.EmitScalarExpr(Arg);
        ptrArgs.push_back(argValue);
      }
    }
    
    // Use external check_ptr() hook to determine aliasing among all pointer
    // arguments. The hook returns true when aliasing EXISTS and false when the
    // pointers are provably independent. We therefore negate its result to get
    // the condition for selecting the noalias version.

    llvm::Value *noAliasCondition;
    if (ptrArgs.size() >= 2) {
      llvm::Type *IntTy = CGF.IntTy;               // "int" in C/IR (i32)

      // Prototype: bool check_ptr(int num_ptrs, ...)
      llvm::FunctionType *CheckPtrTy =
          llvm::FunctionType::get(CGF.Builder.getInt1Ty(), {IntTy}, /*isVarArg=*/true);
      llvm::FunctionCallee CheckPtrFn =
          CGF.CGM.CreateRuntimeFunction(CheckPtrTy, "check_ptr");

      llvm::SmallVector<llvm::Value *, 8> CheckArgs;
      CheckArgs.push_back(
          llvm::ConstantInt::get(IntTy, static_cast<unsigned>(ptrArgs.size())));

      // Cast each argument to void* (i8* in LLVM IR) before passing.
      for (llvm::Value *V : ptrArgs)
        CheckArgs.push_back(CGF.Builder.CreateBitCast(V, CGF.CGM.Int8PtrTy));

      llvm::CallBase *AliasCall =
          CGF.EmitRuntimeCallOrInvoke(CheckPtrFn, CheckArgs);

      // AliasCall == true  => aliasing exists  => use ORIGINAL kernel
      // AliasCall == false => no aliasing      => use NOALIAS kernel
      noAliasCondition = CGF.Builder.CreateNot(AliasCall, "noalias");
    } else {
      // Fewer than two pointer parameters: conservatively assume aliasing and
      // stick to the original kernel implementation.
      noAliasCondition = llvm::ConstantInt::getFalse(CGF.Builder.getContext());
    }

    CGF.Builder.CreateCondBr(noAliasCondition, useNoaliasBlock,
                             useOriginalBlock);
    
    // Noalias branch
    CGF.EmitBlock(useNoaliasBlock);
    // Call the noalias version of the stub function
    if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E->getCallee())) {
      if (const FunctionDecl *CalledFD = dyn_cast<FunctionDecl>(DRE->getDecl())) {
        // Find the noalias version of the stub function
        std::string NoaliasStubName = CGF.CGM.getMangledName(GlobalDecl(CalledFD)).str() + "_noalias";
        if (llvm::Function *NoaliasStub = CGF.CGM.getModule().getFunction(NoaliasStubName)) {
          // Emit the call arguments
          CallArgList Args;
          for (const Expr *Arg : E->arguments()) {
            Args.add(CGF.EmitAnyExpr(Arg), Arg->getType());
          }
          
          // Create function info for the call
          const CGFunctionInfo &FnInfo = CGF.CGM.getTypes().arrangeFreeFunctionCall(
              Args, CalledFD->getType()->castAs<FunctionType>(), /*ChainCall=*/false);
          
          // Emit the call
          CGF.EmitCall(FnInfo, CGCallee::forDirect(NoaliasStub), ReturnValueSlot(), Args);
        } else {
          // Fallback to original call if noalias stub not found
          CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
        }
      } else {
        CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
      }
    } else {
      CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
    }
    CGF.EmitBranch(afterKernelCallBlock);
    
    // Original branch (unused in this case since condition is always true)
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
