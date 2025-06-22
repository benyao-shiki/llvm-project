//===- CudaArgsProfile.cpp - CUDA Kernel Arguments Profiling Pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass instruments CUDA host code to profile kernel launch parameters
// including scalar arguments, blockDim, and gridDim values.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/CudaArgsProfile/CudaArgsProfile.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <map>
#include <string>

using namespace llvm;

#define DEBUG_TYPE "cuda-args-profile"

STATISTIC(CudaKernelLaunches, "Number of CUDA kernel launches instrumented");

namespace {

class CudaArgsProfileImpl {
private:
  // Runtime functions for profiling
  FunctionCallee ProfileKernelLaunch;
  FunctionCallee ProfileScalarArg;
  FunctionCallee ProfileGridDim;
  FunctionCallee ProfileBlockDim;
  
  // Helper functions
  void createProfileFunctions(Module &M);
  bool instrumentCudaLaunchKernel(CallInst *CI, Module &M);
  bool instrumentCudaLaunchKernel_ptsz(CallInst *CI, Module &M);
  void profileKernelArguments(CallInst *CI, Module &M, Value *ArgsArray);
  void profileDimensions(CallInst *CI, Module &M, Value *GridDim, Value *BlockDim);
  std::string extractKernelName(Value *KernelFunc);
  std::string demangleKernelName(const std::string &mangledName);
  
public:
  bool runOnModule(Module &M);
};

void CudaArgsProfileImpl::createProfileFunctions(Module &M) {
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *Int8PtrTy = PointerType::get(Type::getInt8Ty(Ctx), 0);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  
  // void profile_kernel_launch(const char* kernel_name)
  std::vector<Type*> ProfileKernelLaunchArgs = {Int8PtrTy};
  ProfileKernelLaunch = M.getOrInsertFunction(
      "profile_kernel_launch",
      FunctionType::get(VoidTy, ProfileKernelLaunchArgs, false));
  
  // void profile_scalar_arg(const char* arg_name, void* value, int size, int type)
  std::vector<Type*> ProfileScalarArgArgs = {Int8PtrTy, Int8PtrTy, Int32Ty, Int32Ty};
  ProfileScalarArg = M.getOrInsertFunction(
      "profile_scalar_arg",
      FunctionType::get(VoidTy, ProfileScalarArgArgs, false));
  
  // void profile_grid_dim(int x, int y, int z)
  std::vector<Type*> ProfileGridDimArgs = {Int32Ty, Int32Ty, Int32Ty};
  ProfileGridDim = M.getOrInsertFunction(
      "profile_grid_dim",
      FunctionType::get(VoidTy, ProfileGridDimArgs, false));
  
  // void profile_block_dim(int x, int y, int z)
  std::vector<Type*> ProfileBlockDimArgs = {Int32Ty, Int32Ty, Int32Ty};
  ProfileBlockDim = M.getOrInsertFunction(
      "profile_block_dim",
      FunctionType::get(VoidTy, ProfileBlockDimArgs, false));
}

std::string CudaArgsProfileImpl::extractKernelName(Value *KernelFunc) {
  // Handle direct function references
  if (auto *Func = dyn_cast<Function>(KernelFunc)) {
    std::string name = Func->getName().str();
    return demangleKernelName(name);
  }
  
  // Handle GetElementPtr instructions (common in CUDA)
  if (auto *GEP = dyn_cast<GetElementPtrInst>(KernelFunc)) {
    if (auto *GV = dyn_cast<GlobalVariable>(GEP->getPointerOperand())) {
      std::string name = GV->getName().str();
      return demangleKernelName(name);
    }
  }
  
  // Handle casts (very common for kernel function pointers)
  if (auto *Cast = dyn_cast<CastInst>(KernelFunc)) {
    if (auto *Func = dyn_cast<Function>(Cast->getOperand(0))) {
      std::string name = Func->getName().str();
      return demangleKernelName(name);
    }
    if (auto *GV = dyn_cast<GlobalVariable>(Cast->getOperand(0))) {
      std::string name = GV->getName().str();
      return demangleKernelName(name);
    }
  }
  
  // Handle global variable references
  if (auto *GV = dyn_cast<GlobalVariable>(KernelFunc)) {
    std::string name = GV->getName().str();
    return demangleKernelName(name);
  }
  
  // Handle constant expressions
  if (auto *ConstExpr = dyn_cast<ConstantExpr>(KernelFunc)) {
    if (ConstExpr->isCast() && ConstExpr->getNumOperands() > 0) {
      return extractKernelName(ConstExpr->getOperand(0));
    }
  }
  
  return "unknown_kernel";
}

std::string CudaArgsProfileImpl::demangleKernelName(const std::string &mangledName) {
  // Simple demangling for CUDA kernel names
  std::string name = mangledName;
  
  // Remove common CUDA/NVCC prefixes
  if (name.find("__device_stub_") == 0) {
    name = name.substr(14); // Remove "__device_stub_"
  }
  
  // Handle C++ mangled names (basic demangling)
  if (name.find("_Z") == 0) {
    // For CUDA kernels, try to extract the function name
    // Pattern: _Z<len><name>...
    size_t start = 2; // Skip "_Z"
    
    // Extract function name length and name
    if (start < name.length() && std::isdigit(name[start])) {
      size_t len_start = start;
      while (start < name.length() && std::isdigit(name[start])) start++;
      if (len_start < start) {
        int func_len = std::stoi(name.substr(len_start, start - len_start));
        if (start + func_len <= name.length()) {
          std::string func_name = name.substr(start, func_len);
          // Remove __device_stub__ prefix if present
          if (func_name.find("__device_stub__") == 0) {
            func_name = func_name.substr(15);
          }
          return func_name.empty() ? mangledName : func_name;
        }
      }
    }
  }
  
  // Remove template parameters if present
  size_t template_pos = name.find('<');
  if (template_pos != std::string::npos) {
    name = name.substr(0, template_pos);
  }
  
  // Remove namespace qualifiers
  size_t colon_pos = name.rfind("::");
  if (colon_pos != std::string::npos) {
    name = name.substr(colon_pos + 2);
  }
  
  return name.empty() ? mangledName : name;
}

void CudaArgsProfileImpl::profileDimensions(CallInst *CI, Module &M, 
                                           Value *GridDim, Value *BlockDim) {
  IRBuilder<> Builder(CI);
  LLVMContext &Ctx = M.getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  
  // For now, we'll extract basic dimension info
  // In CUDA, dim3 is typically passed as separate values in LLVM IR
  // We'll use placeholder values for demonstration
  Value *GridX = ConstantInt::get(Int32Ty, 1);
  Value *GridY = ConstantInt::get(Int32Ty, 1);
  Value *GridZ = ConstantInt::get(Int32Ty, 1);
  
  Value *BlockX = ConstantInt::get(Int32Ty, 1);
  Value *BlockY = ConstantInt::get(Int32Ty, 1);
  Value *BlockZ = ConstantInt::get(Int32Ty, 1);
  
  // Try to extract actual values if possible
  if (GridDim->getType()->isIntegerTy()) {
    // If it's an integer, use it as X dimension
    if (GridDim->getType()->getIntegerBitWidth() <= 32) {
      GridX = GridDim;
    } else {
      GridX = Builder.CreateTrunc(GridDim, Int32Ty);
    }
  }
  
  if (BlockDim->getType()->isIntegerTy()) {
    // If it's an integer, use it as X dimension
    if (BlockDim->getType()->getIntegerBitWidth() <= 32) {
      BlockX = BlockDim;
    } else {
      BlockX = Builder.CreateTrunc(BlockDim, Int32Ty);
    }
  }
  
  // Call profiling functions
  Builder.CreateCall(ProfileGridDim, {GridX, GridY, GridZ});
  Builder.CreateCall(ProfileBlockDim, {BlockX, BlockY, BlockZ});
}

void CudaArgsProfileImpl::profileKernelArguments(CallInst *CI, Module &M, Value *ArgsArray) {
  IRBuilder<> Builder(CI);
  LLVMContext &Ctx = M.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int8PtrTy = PointerType::get(Int8Ty, 0);
  Type *Int8PtrPtrTy = PointerType::get(Int8PtrTy, 0);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  
  // ArgsArray is void** - we need to iterate through the arguments
  // We'll try to detect scalar arguments vs pointers and only profile scalars
  
  Value *ArgsPtrPtr = Builder.CreateBitCast(ArgsArray, Int8PtrPtrTy);
  
  // Get the kernel function to try to extract parameter information
  Value *KernelFunc = CI->getArgOperand(0);
  Function *KernelFunction = nullptr;
  
  // Try to get the actual kernel function for parameter information
  if (auto *Cast = dyn_cast<CastInst>(KernelFunc)) {
    if (auto *Func = dyn_cast<Function>(Cast->getOperand(0))) {
      KernelFunction = Func;
    }
  } else if (auto *Func = dyn_cast<Function>(KernelFunc)) {
    KernelFunction = Func;
  }
  
  // Profile arguments based on kernel function signature
  // Only process actual arguments, not extra slots
  int numArgs = 4; // Default fallback
  if (KernelFunction) {
    numArgs = std::min((int)KernelFunction->arg_size(), 8);
  }
  
  for (int i = 0; i < numArgs; i++) {
    Value *ArgPtr = Builder.CreateLoad(Int8PtrTy, 
                                      Builder.CreateGEP(Int8PtrTy, ArgsPtrPtr, 
                                                       ConstantInt::get(Int32Ty, i)));
    
    // Get argument name and try to determine if it's a scalar
    std::string ArgName;
    bool isLikelyScalar = false;
    
    if (KernelFunction && i < KernelFunction->arg_size()) {
      auto ArgIter = KernelFunction->arg_begin();
      std::advance(ArgIter, i);
      if (ArgIter->hasName()) {
        ArgName = ArgIter->getName().str();
      } else {
        ArgName = "param_" + std::to_string(i);
      }
      
      // Check if argument type suggests it's a scalar
      Type *ArgType = ArgIter->getType();
      if (ArgType->isIntegerTy() || ArgType->isFloatingPointTy()) {
        isLikelyScalar = true;
      } else if (ArgType->isPointerTy()) {
        // Skip pointer arguments
        continue;
      }
    } else {
      ArgName = "param_" + std::to_string(i);
      isLikelyScalar = true; // Assume scalar for unknown args
    }
    
    // Only profile if it's likely a scalar
    if (isLikelyScalar) {
      Value *ArgNameStr = Builder.CreateGlobalString(ArgName);
      Value *ArgIndex = ConstantInt::get(Int32Ty, i);
      
      Builder.CreateCall(ProfileScalarArg, {ArgNameStr, ArgPtr, ArgIndex, ArgIndex});
    }
  }
}

bool CudaArgsProfileImpl::instrumentCudaLaunchKernel(CallInst *CI, Module &M) {
  // cudaLaunchKernel in LLVM IR: (ptr, i64, i32, i64, i32, ptr, i64, ptr)
  // func, gridDim.x+y, gridDim.z, blockDim.x+y, blockDim.z, args, sharedMem, stream
  if (CI->getNumOperands() < 8) return false; // +1 for the function being called
  
  IRBuilder<> Builder(CI);
  
  // Extract kernel name and profile kernel launch
  Value *KernelFunc = CI->getArgOperand(0);
  std::string KernelName = extractKernelName(KernelFunc);
  Value *KernelNameStr = Builder.CreateGlobalString(KernelName);
  Builder.CreateCall(ProfileKernelLaunch, {KernelNameStr});
  
  // Profile dimensions - use the simplified approach for now
  Value *GridDimXY = CI->getArgOperand(1);   // i64 containing x and y
  Value *BlockDimXY = CI->getArgOperand(3);  // i64 containing x and y
  profileDimensions(CI, M, GridDimXY, BlockDimXY);
  
  // Profile kernel arguments
  Value *ArgsArray = CI->getArgOperand(5); // args array
  profileKernelArguments(CI, M, ArgsArray);
  
  ++CudaKernelLaunches;
  return true;
}

bool CudaArgsProfileImpl::instrumentCudaLaunchKernel_ptsz(CallInst *CI, Module &M) {
  // Same as cudaLaunchKernel but with per-thread stream
  return instrumentCudaLaunchKernel(CI, M);
}

bool CudaArgsProfileImpl::runOnModule(Module &M) {
  bool Modified = false;
  
  createProfileFunctions(M);
  
  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          if (Function *CalledFunc = CI->getCalledFunction()) {
            StringRef FuncName = CalledFunc->getName();
            
            if (FuncName == "cudaLaunchKernel" || FuncName == "hipLaunchKernel") {
              Modified |= instrumentCudaLaunchKernel(CI, M);
            } else if (FuncName == "cudaLaunchKernel_ptsz" || FuncName == "hipLaunchKernel_spt") {
              Modified |= instrumentCudaLaunchKernel_ptsz(CI, M);
            }
          }
        }
      }
    }
  }
  
  return Modified;
}

} // anonymous namespace

PreservedAnalyses CudaArgsProfilePass::run(Module &M, ModuleAnalysisManager &AM) {
  CudaArgsProfileImpl Impl;
  bool Modified = Impl.runOnModule(M);
  return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
} 