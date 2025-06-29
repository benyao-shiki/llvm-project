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
    
    // Generate runtime condition check
    const CallExpr *CallE = cast<CallExpr>(E);
    
    // Collect pointer arguments
    llvm::SmallVector<llvm::Value *, 4> ptrArgs;
    for (unsigned i = 0; i < CallE->getNumArgs(); ++i) {
      const Expr *Arg = CallE->getArg(i);
      if (Arg->getType()->isPointerType()) {
        llvm::Value *argValue = CGF.EmitScalarExpr(Arg);
        ptrArgs.push_back(argValue);
      }
    }
    
    // Runtime condition: if we have at least 2 pointer args AND first != second, use noalias
    llvm::Value *condition;
    if (ptrArgs.size() >= 2) {
      // Compare first and second pointer addresses
      llvm::Value *firstPtr = ptrArgs[0];
      llvm::Value *secondPtr = ptrArgs[1];
      
      // Convert pointers to integers for comparison
      llvm::Value *firstPtrInt = CGF.Builder.CreatePtrToInt(firstPtr, CGF.Int64Ty, "first_ptr_int");
      llvm::Value *secondPtrInt = CGF.Builder.CreatePtrToInt(secondPtr, CGF.Int64Ty, "second_ptr_int");
      
      // Check if addresses are different (no aliasing)
      condition = CGF.Builder.CreateICmpNE(firstPtrInt, secondPtrInt, "ptrs_different");
    } else {
      // Less than 2 pointer args, don't use noalias
      condition = CGF.Builder.getFalse();
    }
    CGF.Builder.CreateCondBr(condition, useNoaliasBlock, useOriginalBlock);
    
    // Noalias branch
    CGF.EmitBlock(useNoaliasBlock);
    // Call the noalias version of the stub function
    if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(CallE->getCallee())) {
      if (const FunctionDecl *CalledFD = dyn_cast<FunctionDecl>(DRE->getDecl())) {
        // Find the noalias version of the stub function
        std::string NoaliasStubName = CGF.CGM.getMangledName(GlobalDecl(CalledFD)).str() + "_noalias";
        if (llvm::Function *NoaliasStub = CGF.CGM.getModule().getFunction(NoaliasStubName)) {
          // Emit the call arguments
          CallArgList Args;
          for (const Expr *Arg : CallE->arguments()) {
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
