//===- CudaArgsProfile.cpp - CUDA Kernel Arguments Profiling Pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass instruments CUDA host code to profile kernel launch parameters.
// It relies on type information injected by Clang into the module.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/CudaArgsProfile/CudaArgsProfile.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <string>
#include <vector>

using namespace llvm;

namespace {

class CudaArgsProfileImpl {
private:
  Module *M;
  LLVMContext *Context;

  // Runtime functions
  Function *ProfileStartFunc;
  Function *ProfileArgumentFunc;
  Function *ProfileEndFunc;
  Function *ProfileFinalizeFunc;

public:
  CudaArgsProfileImpl(Module &Mod) : M(&Mod), Context(&Mod.getContext()) {
    createRuntimeFunctions();
  }

  bool runOnModule();

private:
  void createRuntimeFunctions();
  bool instrumentCudaLaunchCalls();
  bool isCudaLaunchCall(CallInst *CI);
  void insertProfilingCalls(CallInst *LaunchCall);
  void createModuleFinalizer();
};

void CudaArgsProfileImpl::createRuntimeFunctions() {
  Type *VoidTy = Type::getVoidTy(*Context);
  Type *Int32Ty = Type::getInt32Ty(*Context);
  Type *CharPtrTy = PointerType::getUnqual(*Context);
  Type *VoidPtrTy = PointerType::getUnqual(*Context);

  // void __cuda_profile_kernel_start(const char* kernel_name, int gX, int gY, int gZ, int bX, int bY, int bZ)
  ProfileStartFunc = M->getFunction("__cuda_profile_kernel_start");
  if (!ProfileStartFunc) {
    ProfileStartFunc = Function::Create(
        FunctionType::get(VoidTy, {CharPtrTy, Int32Ty, Int32Ty, Int32Ty, Int32Ty, Int32Ty, Int32Ty}, false),
        Function::ExternalLinkage, "__cuda_profile_kernel_start", M);
  }

  // void __cuda_profile_argument(int index, void* value_ptr)
  ProfileArgumentFunc = M->getFunction("__cuda_profile_argument");
  if (!ProfileArgumentFunc) {
    ProfileArgumentFunc = Function::Create(
        FunctionType::get(VoidTy, {Int32Ty, VoidPtrTy}, false),
        Function::ExternalLinkage, "__cuda_profile_argument", M);
  }

  // void __cuda_profile_kernel_end()
  ProfileEndFunc = M->getFunction("__cuda_profile_kernel_end");
  if (!ProfileEndFunc) {
    ProfileEndFunc = Function::Create(
        FunctionType::get(VoidTy, {}, false),
        Function::ExternalLinkage, "__cuda_profile_kernel_end", M);
  }

  // void __cuda_profile_finalize()
  ProfileFinalizeFunc = M->getFunction("__cuda_profile_finalize");
  if (!ProfileFinalizeFunc) {
    ProfileFinalizeFunc = Function::Create(
        FunctionType::get(VoidTy, {}, false),
        Function::ExternalLinkage, "__cuda_profile_finalize", M);
  }
}

bool CudaArgsProfileImpl::runOnModule() {
  // This pass is disabled. The instrumentation is now done in Clang's CodeGen.
  return false;
}

bool CudaArgsProfileImpl::isCudaLaunchCall(CallInst *CI) {
  Function *Callee = CI->getCalledFunction();
  if (!Callee) return false;
  StringRef Name = Callee->getName();
  // Check for standard CUDA launch functions.
  return Name == "cudaLaunchKernel";
}

void CudaArgsProfileImpl::insertProfilingCalls(CallInst *LaunchCall) {
  IRBuilder<> Builder(LaunchCall);
  Type *Int32Ty = Type::getInt32Ty(*Context);

  // 1. Get kernel name from the first argument of the launch call.
  Value *KernelFuncPtr = LaunchCall->getArgOperand(0);
  // The pointer might be casted, so strip casts to get the underlying global variable (the function).
  KernelFuncPtr = KernelFuncPtr->stripPointerCasts();
  Value *KernelName = Builder.CreateGlobalStringPtr(KernelFuncPtr->getName(), "kernel_name");

  // 2. Get grid and block dimensions from the call to __cudaPopCallConfiguration
  // This requires finding the preceding call.
  Value *GridX = Builder.getInt32(1), *GridY = Builder.getInt32(1), *GridZ = Builder.getInt32(1);
  Value *BlockX = Builder.getInt32(1), *BlockY = Builder.getInt32(1), *BlockZ = Builder.getInt32(1);

  if (CallInst *PopCall = dyn_cast<CallInst>(LaunchCall->getPrevNode())) {
      if (PopCall->getCalledFunction() && PopCall->getCalledFunction()->getName().contains("PopCallConfiguration")) {
          Value *GridDimPtr = PopCall->getArgOperand(0);
          Value *BlockDimPtr = PopCall->getArgOperand(1);

          Type *Dim3Ty = cast<PointerType>(GridDimPtr->getType())->getContainedType(0);

          GridX = Builder.CreateLoad(Int32Ty, Builder.CreateStructGEP(Dim3Ty, GridDimPtr, 0));
          GridY = Builder.CreateLoad(Int32Ty, Builder.CreateStructGEP(Dim3Ty, GridDimPtr, 1));
          GridZ = Builder.CreateLoad(Int32Ty, Builder.CreateStructGEP(Dim3Ty, GridDimPtr, 2));

          BlockX = Builder.CreateLoad(Int32Ty, Builder.CreateStructGEP(Dim3Ty, BlockDimPtr, 0));
          BlockY = Builder.CreateLoad(Int32Ty, Builder.CreateStructGEP(Dim3Ty, BlockDimPtr, 1));
          BlockZ = Builder.CreateLoad(Int32Ty, Builder.CreateStructGEP(Dim3Ty, BlockDimPtr, 2));
      }
  }

  // 3. Call __cuda_profile_kernel_start
  Builder.CreateCall(ProfileStartFunc, {KernelName, GridX, GridY, GridZ, BlockX, BlockY, BlockZ});

  // 4. Loop through arguments and call __cuda_profile_argument
  Value *KernelOperand = LaunchCall->getArgOperand(0)->stripPointerCasts();
  Function *KernelFunc = dyn_cast<Function>(KernelOperand);
  if (!KernelFunc) {
    if (auto *GV = dyn_cast<GlobalVariable>(KernelOperand)) {
      if (GV->hasInitializer()) {
        KernelFunc = dyn_cast<Function>(GV->getInitializer()->stripPointerCasts());
      }
    }
  }

  if (KernelFunc) {
      unsigned numParams = KernelFunc->getFunctionType()->getNumParams();
      Value *ArgsArray = LaunchCall->getArgOperand(3);
      Type *VoidPtrTy = PointerType::getUnqual(*Context);

      for (unsigned i = 0; i < numParams; ++i) {
        Value *Idx = Builder.getInt32(i);
        Value *ArgPtrPtr = Builder.CreateGEP(VoidPtrTy, ArgsArray, Idx, "arg_addr");
        Value *ArgPtr = Builder.CreateLoad(VoidPtrTy, ArgPtrPtr, "arg_ptr");
        Builder.CreateCall(ProfileArgumentFunc, {Idx, ArgPtr});
      }
  }

  // 5. Call __cuda_profile_kernel_end
  Builder.CreateCall(ProfileEndFunc, {});
}

bool CudaArgsProfileImpl::instrumentCudaLaunchCalls() {
  bool Changed = false;
  std::vector<CallInst *> LaunchCalls;

  for (Function &F : *M) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          if (isCudaLaunchCall(CI)) {
            LaunchCalls.push_back(CI);
          }
        }
      }
    }
  }

  for (CallInst *CI : LaunchCalls) {
    insertProfilingCalls(CI);
    Changed = true;
  }

  return Changed;
}

void CudaArgsProfileImpl::createModuleFinalizer() {
  Function *FiniFunc = Function::Create(
    FunctionType::get(Type::getVoidTy(*Context), {}, false),
    Function::InternalLinkage, "__cuda_profile_fini", M);

  BasicBlock *Entry = BasicBlock::Create(*Context, "entry", FiniFunc);
  IRBuilder<> Builder(Entry);
  Builder.CreateCall(ProfileFinalizeFunc, {});
  Builder.CreateRetVoid();

  appendToGlobalDtors(*M, FiniFunc, 0);
}

} // anonymous namespace

PreservedAnalyses CudaArgsProfilePass::run(Module &M, ModuleAnalysisManager &AM) {
  CudaArgsProfileImpl Impl(M);
  if (Impl.runOnModule()) {
    return PreservedAnalyses::none();
  }
  return PreservedAnalyses::all();
}
