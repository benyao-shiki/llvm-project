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
#include <functional>

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
  // 递归提取kernel名字的通用方法
  std::function<std::string(Value*, int)> extractRecursive = [&](Value* V, int depth) -> std::string {
    if (depth > 10) return "deep_nested_kernel"; // 防止无限递归
    
    // 1. 直接函数引用
    if (auto *Func = dyn_cast<Function>(V)) {
      return demangleKernelName(Func->getName().str());
    }
    
    // 2. 全局变量（kernel stub）
    if (auto *GV = dyn_cast<GlobalVariable>(V)) {
      return demangleKernelName(GV->getName().str());
    }
    
    // 3. 各种指令类型的通用处理
    if (auto *Inst = dyn_cast<Instruction>(V)) {
      switch (Inst->getOpcode()) {
        case Instruction::BitCast:
        case Instruction::AddrSpaceCast:
        case Instruction::IntToPtr:
        case Instruction::PtrToInt:
          // 透过类型转换继续查找
          if (Inst->getNumOperands() > 0) {
            return extractRecursive(Inst->getOperand(0), depth + 1);
          }
          break;
          
        case Instruction::GetElementPtr:
          // GEP通常指向全局数组中的函数指针
          if (Inst->getNumOperands() > 0) {
            return extractRecursive(Inst->getOperand(0), depth + 1);
          }
          break;
          
        case Instruction::Load:
          // 从内存加载的函数指针
          if (auto *Load = dyn_cast<LoadInst>(Inst)) {
            std::string loaded_name = extractRecursive(Load->getPointerOperand(), depth + 1);
            if (loaded_name != "unknown_kernel") {
              return loaded_name + "_loaded";
            }
          }
          break;
          
        case Instruction::PHI:
          // 条件选择的kernel
          if (auto *PHI = dyn_cast<PHINode>(Inst)) {
            for (unsigned i = 0; i < PHI->getNumIncomingValues(); ++i) {
              std::string name = extractRecursive(PHI->getIncomingValue(i), depth + 1);
              if (name != "unknown_kernel") {
                return name + "_variant" + std::to_string(i);
              }
            }
          }
          break;
          
        case Instruction::Select:
          // 条件选择
          if (auto *Select = dyn_cast<SelectInst>(Inst)) {
            std::string trueName = extractRecursive(Select->getTrueValue(), depth + 1);
            std::string falseName = extractRecursive(Select->getFalseValue(), depth + 1);
            if (trueName != "unknown_kernel") return trueName + "_true";
            if (falseName != "unknown_kernel") return falseName + "_false";
          }
          break;
          
        default:
          break;
      }
    }
    
    // 4. 常量表达式的通用处理
    if (auto *ConstExpr = dyn_cast<ConstantExpr>(V)) {
      if (ConstExpr->getNumOperands() > 0) {
        return extractRecursive(ConstExpr->getOperand(0), depth + 1);
      }
    }
    
    // 5. 参数和其他值类型
    if (auto *Arg = dyn_cast<Argument>(V)) {
      return "param_kernel_" + std::to_string(Arg->getArgNo());
    }
    
    return "unknown_kernel";
  };
  
  return extractRecursive(KernelFunc, 0);
}

std::string CudaArgsProfileImpl::demangleKernelName(const std::string &mangledName) {
  std::string name = mangledName;
  
  // 通用的CUDA/GPU相关前缀清理
  std::vector<std::string> prefixes_to_remove = {
    "__device_stub_",
    "__global_stub_", 
    "__cuda_",
    "__hip_",
    "_kernel_stub_",
    "__kernel_",
    "kernel_stub_"
  };
  
  for (const auto &prefix : prefixes_to_remove) {
    if (name.find(prefix) == 0) {
      name = name.substr(prefix.length());
      break; // 只移除第一个匹配的前缀
    }
  }
  
  // 通用的C++符号处理
  if (name.find("_Z") == 0) {
    // 标准C++ name mangling
    size_t start = 2;
    if (start < name.length() && std::isdigit(name[start])) {
      size_t len_start = start;
      while (start < name.length() && std::isdigit(name[start])) start++;
      if (len_start < start) {
        try {
          int func_len = std::stoi(name.substr(len_start, start - len_start));
          if (start + func_len <= name.length()) {
            name = name.substr(start, func_len);
          }
        } catch (...) {
          // 保持原名
        }
      }
    }
  }
  
  // 移除通用的后缀
  std::vector<std::string> suffixes_to_remove = {
    "_stub",
    "_wrapper", 
    "_impl",
    "_kernel"
  };
  
  for (const auto &suffix : suffixes_to_remove) {
    size_t pos = name.rfind(suffix);
    if (pos != std::string::npos && pos + suffix.length() == name.length()) {
      name = name.substr(0, pos);
      break;
    }
  }
  
  // 清理模板参数
  size_t template_pos = name.find('<');
  if (template_pos != std::string::npos) {
    name = name.substr(0, template_pos);
  }
  
  // 保留最后一个命名空间组件
  size_t colon_pos = name.rfind("::");
  if (colon_pos != std::string::npos) {
    name = name.substr(colon_pos + 2);
  }
  
  // 清理前导下划线（但保留有意义的名字）
  while (name.length() > 1 && name[0] == '_' && name[1] != '_') {
    name = name.substr(1);
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
  
  Value *ArgsPtrPtr = Builder.CreateBitCast(ArgsArray, Int8PtrPtrTy);
  
  // 通用的kernel函数查找策略
  Value *KernelFunc = CI->getArgOperand(0);
  Function *KernelFunction = nullptr;
  
  // 递归查找kernel函数的通用方法
  std::function<Function*(Value*, int)> findKernelFunction = [&](Value* V, int depth) -> Function* {
    if (depth > 5) return nullptr;
    
    if (auto *Func = dyn_cast<Function>(V)) {
      return Func;
    }
    
    if (auto *Cast = dyn_cast<CastInst>(V)) {
      if (Cast->getNumOperands() > 0) {
        return findKernelFunction(Cast->getOperand(0), depth + 1);
      }
    }
    
    if (auto *ConstExpr = dyn_cast<ConstantExpr>(V)) {
      if (ConstExpr->isCast() && ConstExpr->getNumOperands() > 0) {
        return findKernelFunction(ConstExpr->getOperand(0), depth + 1);
      }
    }
    
    if (auto *Load = dyn_cast<LoadInst>(V)) {
      if (auto *GV = dyn_cast<GlobalVariable>(Load->getPointerOperand())) {
        // 尝试通过全局变量名找到对应的函数
        std::string globalName = GV->getName().str();
        
        // 通用的stub到函数名映射
        std::vector<std::pair<std::string, std::string>> stub_patterns = {
          {"__device_stub_", ""},
          {"_kernel_stub_", ""},
          {"__global_stub_", ""},
          {"_stub_", ""},
        };
        
        for (const auto &pattern : stub_patterns) {
          if (globalName.find(pattern.first) == 0) {
            std::string funcName = globalName.substr(pattern.first.length()) + pattern.second;
            if (Function *F = M.getFunction(funcName)) {
              return F;
            }
          }
        }
      }
    }
    
    return nullptr;
  };
  
  KernelFunction = findKernelFunction(KernelFunc, 0);
  
  // 智能参数数量推断
  int numArgs = 8; // 保守的默认值
  if (KernelFunction) {
    numArgs = std::min((int)KernelFunction->arg_size(), 16); // 增加上限
  } else {
    // 如果找不到kernel函数，尝试从调用上下文推断
    // 检查是否有明显的参数数量线索
    if (CI->getNumOperands() >= 6) { // cudaLaunchKernel的参数数量
      // 可以尝试从其他参数推断，但现在保持保守
      numArgs = 6;
    }
  }
  
  for (int i = 0; i < numArgs; i++) {
    Value *ArgPtr = Builder.CreateLoad(Int8PtrTy, 
                                      Builder.CreateGEP(Int8PtrTy, ArgsPtrPtr, 
                                                       ConstantInt::get(Int32Ty, i)));
    
    // 通用的参数名称和类型推断
    std::string ArgName = "param_" + std::to_string(i);
    bool isLikelyScalar = false;
    int typeHint = 0; // 0=unknown, 1-64=bit width, 100=float, 200=double
    
    if (KernelFunction && i < KernelFunction->arg_size()) {
      auto ArgIter = KernelFunction->arg_begin();
      std::advance(ArgIter, i);
      
      // 获取参数名（如果有的话）
      if (ArgIter->hasName()) {
        ArgName = ArgIter->getName().str();
      }
      
      // 通用的标量类型检测
      Type *ArgType = ArgIter->getType();
      if (ArgType->isIntegerTy()) {
        isLikelyScalar = true;
        typeHint = ArgType->getIntegerBitWidth();
      } else if (ArgType->isFloatingPointTy()) {
        isLikelyScalar = true;
        if (ArgType->isFloatTy()) {
          typeHint = 100; // float
        } else if (ArgType->isDoubleTy()) {
          typeHint = 200; // double
        } else {
          typeHint = 100; // 其他浮点类型默认为float
        }
      } else if (ArgType->isPointerTy()) {
        // 指针类型通常是设备内存，跳过
        continue;
      } else {
        // 其他类型（结构体等）可能包含标量，保守处理
        isLikelyScalar = true;
        typeHint = 32; // 默认32位
      }
    } else {
      // 没有类型信息时的启发式判断
      isLikelyScalar = true;
      typeHint = 32; // 默认32位
    }
    
    // 只profile可能的标量参数
    if (isLikelyScalar) {
      Value *ArgNameStr = Builder.CreateGlobalString(ArgName);
      Value *ArgIndex = ConstantInt::get(Int32Ty, i);
      Value *TypeInfo = ConstantInt::get(Int32Ty, typeHint);
      
      Builder.CreateCall(ProfileScalarArg, {ArgNameStr, ArgPtr, ArgIndex, TypeInfo});
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
            
            // 标准CUDA Runtime API
            if (FuncName == "cudaLaunchKernel" || FuncName == "hipLaunchKernel") {
              Modified |= instrumentCudaLaunchKernel(CI, M);
            } else if (FuncName == "cudaLaunchKernel_ptsz" || FuncName == "hipLaunchKernel_spt") {
              Modified |= instrumentCudaLaunchKernel_ptsz(CI, M);
            }
            // CUDA Driver API
            else if (FuncName == "cuLaunchKernel") {
              // TODO: 可以在将来添加cuLaunchKernel支持
              LLVM_DEBUG(dbgs() << "Found cuLaunchKernel - not implemented yet\n");
            }
          } else {
            // 间接调用 - 可能是函数指针
            // 在调试模式下记录这些调用以供分析
            LLVM_DEBUG(dbgs() << "Indirect call found: " << *CI << "\n");
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