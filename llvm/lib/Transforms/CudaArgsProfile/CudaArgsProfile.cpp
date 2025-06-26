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
// Enhanced version with debug info support for preserving parameter names.
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
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <map>
#include <string>
#include <functional>
#include <unordered_map>
#include <regex>

using namespace llvm;

#define DEBUG_TYPE "cuda-args-profile"

STATISTIC(CudaKernelLaunches, "Number of CUDA kernel launches instrumented");

namespace {

// 参数信息结构
struct ParamInfo {
  std::string name;
  std::string type;
  int size;
  bool isScalar;
  int typeHint; // 0=unknown, 1-64=bit width, 100=float, 200=double
};

// 全局的kernel参数信息映射
static std::unordered_map<std::string, std::vector<ParamInfo>> KernelParamCache;

class CudaArgsProfileImpl {
private:
  // Runtime functions for profiling
  FunctionCallee ProfileKernelLaunch;
  FunctionCallee ProfileScalarArg;
  FunctionCallee ProfileGridDim;
  FunctionCallee ProfileBlockDim;
  FunctionCallee RegisterKernelParams;
  
  // Helper functions
  void createProfileFunctions(Module &M);
  bool instrumentCudaLaunchKernel(CallInst *CI, Module &M);
  bool instrumentCudaLaunchKernel_ptsz(CallInst *CI, Module &M);
  void profileKernelArguments(CallInst *CI, Module &M, Value *ArgsArray, const std::string &kernelName);
  void profileDimensions(CallInst *CI, Module &M, Value *GridDim, Value *BlockDim);
  std::string extractKernelName(Value *KernelFunc);
  std::string demangleKernelName(const std::string &mangledName);
  
  // 新增：调试信息和参数名称相关方法
  void extractDebugParamInfo(Module &M);
  std::vector<ParamInfo> getKernelParamInfo(const std::string &kernelName, Function *KernelFunc);
  std::vector<ParamInfo> extractParamInfoFromDebug(Function *F);
  std::vector<ParamInfo> extractParamInfoFromHeuristics(Function *F, const std::string &kernelName);
  void registerKernelParametersAtRuntime(Module &M);
  std::string inferTypeFromLLVMType(Type *T);
  
public:
  bool runOnModule(Module &M);
};

void CudaArgsProfileImpl::createProfileFunctions(Module &M) {
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *Int8PtrTy = PointerType::get(Ctx, 0);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  
  // void profile_kernel_launch(const char* kernel_name)
  std::vector<Type*> ProfileKernelLaunchArgs = {Int8PtrTy};
  ProfileKernelLaunch = M.getOrInsertFunction(
      "profile_kernel_launch",
      FunctionType::get(VoidTy, ProfileKernelLaunchArgs, false));
  
  // void profile_scalar_arg(const char* arg_name, int arg_index, void* value_ptr, int type_info, const char* type_name)
  std::vector<Type*> ProfileScalarArgArgs = {Int8PtrTy, Int32Ty, Int8PtrTy, Int32Ty, Int8PtrTy};
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
  
  // void register_kernel_params(const char* kernel_name, int param_count, const char** param_names, const char** param_types)
  std::vector<Type*> RegisterKernelParamsArgs = {Int8PtrTy, Int32Ty, PointerType::get(Ctx, 0), PointerType::get(Ctx, 0)};
  RegisterKernelParams = M.getOrInsertFunction(
      "register_kernel_params",
      FunctionType::get(VoidTy, RegisterKernelParamsArgs, false));
}

// 从调试信息中提取参数信息
void CudaArgsProfileImpl::extractDebugParamInfo(Module &M) {
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    
    // 检查是否是CUDA kernel（包括device stub函数）
    StringRef FuncName = F.getName();
    if (F.getCallingConv() == CallingConv::PTX_Kernel ||
        FuncName.contains("kernel") ||
        FuncName.contains("__device_stub__") ||
        FuncName.contains("__global_stub__") ||
        F.getMetadata("kernel") != nullptr) {
      
      std::vector<ParamInfo> params = extractParamInfoFromDebug(&F);
      if (!params.empty()) {
        std::string kernelName = demangleKernelName(F.getName().str());
        KernelParamCache[kernelName] = params;
        KernelParamCache[F.getName().str()] = params; // 也保存原名
        
        // 添加调试输出
        LLVM_DEBUG(dbgs() << "提取到kernel调试信息: " << kernelName 
                          << ", 参数数量: " << params.size() << "\n");
        for (size_t i = 0; i < params.size(); i++) {
          LLVM_DEBUG(dbgs() << "  参数" << i << ": " << params[i].name 
                            << " (" << params[i].type << ")\n");
        }
      }
    }
  }
}

std::string CudaArgsProfileImpl::inferTypeFromLLVMType(Type *T) {
  if (T->isFloatTy()) return "float";
  if (T->isDoubleTy()) return "double";
  if (T->isIntegerTy()) {
    unsigned bitWidth = T->getIntegerBitWidth();
    return "int" + std::to_string(bitWidth);
  }
  if (T->isPointerTy()) {
    return "pointer";
  }
  return "unknown";
}

std::vector<ParamInfo> CudaArgsProfileImpl::extractParamInfoFromDebug(Function *F) {
  std::vector<ParamInfo> params;
  
  // 尝试从调试信息获取
  if (DISubprogram *SP = F->getSubprogram()) {
    // 首先尝试从DISubprogram的局部变量中获取参数信息
    unsigned ParamIndex = 0;
    for (Argument &Arg : F->args()) {
      ParamInfo param;
      bool foundDebugInfo = false;
      
      // 查找对应的DILocalVariable
      for (const DINode *Node : SP->getRetainedNodes()) {
        if (const DILocalVariable *Var = dyn_cast<DILocalVariable>(Node)) {
          if (Var->isParameter() && Var->getArg() == ParamIndex + 1) {
            param.name = Var->getName().str();
            if (DIType *VarType = Var->getType()) {
              param.type = VarType->getName().str();
            }
            foundDebugInfo = true;
            break;
          }
        }
      }
      
      // 如果没有找到调试信息，使用参数本身的名字或默认名
      if (!foundDebugInfo) {
        if (Arg.hasName()) {
          param.name = Arg.getName().str();
        } else {
          param.name = "param_" + std::to_string(ParamIndex);
        }
      }
      
      // 分析LLVM类型特征
      Type *LLVMType = Arg.getType();
      if (LLVMType->isIntegerTy()) {
        param.isScalar = true;
        param.typeHint = LLVMType->getIntegerBitWidth();
        param.size = (param.typeHint + 7) / 8;
        if (param.type.empty()) {
          param.type = "int" + std::to_string(param.typeHint);
        }
      } else if (LLVMType->isFloatingPointTy()) {
        param.isScalar = true;
        if (LLVMType->isFloatTy()) {
          param.typeHint = 100;
          param.size = 4;
          if (param.type.empty()) param.type = "float";
        } else if (LLVMType->isDoubleTy()) {
          param.typeHint = 200;
          param.size = 8;
          if (param.type.empty()) param.type = "double";
        } else if (LLVMType->isHalfTy()) {
          param.typeHint = 50;
          param.size = 2;
          if (param.type.empty()) param.type = "half";
        }
      } else if (LLVMType->isPointerTy()) {
        param.isScalar = false;
        param.typeHint = 0;
        param.size = 8; // pointer size
        if (param.type.empty()) param.type = "pointer";
      } else {
        param.isScalar = false;
        param.typeHint = 0;
        param.size = 0;
        if (param.type.empty()) param.type = "unknown";
      }
      
      params.push_back(param);
      ParamIndex++;
    }
  }
  
  // 如果调试信息不可用，使用启发式方法
  if (params.empty()) {
    params = extractParamInfoFromHeuristics(F, F->getName().str());
  }
  
  return params;
}

std::vector<ParamInfo> CudaArgsProfileImpl::extractParamInfoFromHeuristics(Function *F, const std::string &kernelName) {
  std::vector<ParamInfo> params;
  
  // 通用参数推断 - 不针对特定kernel类型做特殊处理
  unsigned argIndex = 0;
  for (Argument &Arg : F->args()) {
    ParamInfo param;
    
    // 优先使用参数本身的名称（如果存在）
    if (Arg.hasName()) {
      param.name = Arg.getName().str();
    } else {
      // 基于参数位置和类型的智能命名
      Type *ArgType = Arg.getType();
      if (ArgType->isPointerTy()) {
        param.name = "ptr_" + std::to_string(argIndex);
      } else if (ArgType->isFloatingPointTy()) {
        param.name = "scalar_" + std::to_string(argIndex);
      } else if (ArgType->isIntegerTy()) {
        param.name = "int_" + std::to_string(argIndex);
      } else {
        param.name = "param_" + std::to_string(argIndex);
      }
    }
    
    param.type = inferTypeFromLLVMType(Arg.getType());
    
    Type *ArgType = Arg.getType();
    if (ArgType->isPointerTy()) {
      param.isScalar = false;
      param.size = 8;
      param.typeHint = 0;
    } else if (ArgType->isFloatingPointTy()) {
      param.isScalar = true;
      if (ArgType->isFloatTy()) {
        param.typeHint = 100;
        param.size = 4;
      } else if (ArgType->isDoubleTy()) {
        param.typeHint = 200;
        param.size = 8;
      } else if (ArgType->isHalfTy()) {
        param.typeHint = 50; // 新增：半精度浮点
        param.size = 2;
      }
    } else if (ArgType->isIntegerTy()) {
      param.isScalar = true;
      param.typeHint = ArgType->getIntegerBitWidth();
      param.size = (param.typeHint + 7) / 8;
    } else {
      // 其他类型（结构体等）
      param.isScalar = false;
      param.typeHint = 0;
      param.size = 0; // 未知大小
    }
    
    params.push_back(param);
    argIndex++;
  }
  
  return params;
}

std::vector<ParamInfo> CudaArgsProfileImpl::getKernelParamInfo(const std::string &kernelName, Function *KernelFunc) {
  // 首先查找缓存
  auto it = KernelParamCache.find(kernelName);
  if (it != KernelParamCache.end()) {
    return it->second;
  }
  
  // 如果有函数定义，从中提取
  if (KernelFunc) {
    std::vector<ParamInfo> params = extractParamInfoFromDebug(KernelFunc);
    if (!params.empty()) {
      KernelParamCache[kernelName] = params;
      return params;
    }
  }
  
  // 使用启发式方法
  if (KernelFunc) {
    std::vector<ParamInfo> params = extractParamInfoFromHeuristics(KernelFunc, kernelName);
    KernelParamCache[kernelName] = params;
    return params;
  }
  
  return {};
}

void CudaArgsProfileImpl::registerKernelParametersAtRuntime(Module &M) {
  // 创建全局构造函数来注册参数信息
  LLVMContext &Ctx = M.getContext();
  FunctionType *InitFuncType = FunctionType::get(Type::getVoidTy(Ctx), false);
  Function *InitFunc = Function::Create(InitFuncType, GlobalValue::InternalLinkage, 
                                       "cuda_profile_register_params", &M);
  
  BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", InitFunc);
  IRBuilder<> Builder(EntryBB);
  
  Type *Int8PtrTy = PointerType::get(Ctx, 0);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  
  for (const auto &kernelInfo : KernelParamCache) {
    const std::string &kernelName = kernelInfo.first;
    const std::vector<ParamInfo> &params = kernelInfo.second;
    
    if (params.empty()) continue;
    
    // 创建参数名和类型的字符串数组
    std::vector<Constant*> paramNames;
    std::vector<Constant*> paramTypes;
    
    for (const auto &param : params) {
      paramNames.push_back(Builder.CreateGlobalString(param.name));
      paramTypes.push_back(Builder.CreateGlobalString(param.type));
    }
    
    ArrayType *StringPtrArrayType = ArrayType::get(Int8PtrTy, params.size());
    
    GlobalVariable *NamesArray = new GlobalVariable(
        M, StringPtrArrayType, true, GlobalValue::PrivateLinkage,
        ConstantArray::get(StringPtrArrayType, paramNames),
        kernelName + "_param_names");
    
    GlobalVariable *TypesArray = new GlobalVariable(
        M, StringPtrArrayType, true, GlobalValue::PrivateLinkage,
        ConstantArray::get(StringPtrArrayType, paramTypes),
        kernelName + "_param_types");
    
    Value *KernelNameStr = Builder.CreateGlobalString(kernelName);
    Value *ParamCount = ConstantInt::get(Int32Ty, params.size());
    
    Value *NamesPtr = Builder.CreateBitCast(NamesArray, PointerType::get(Ctx, 0));
    Value *TypesPtr = Builder.CreateBitCast(TypesArray, PointerType::get(Ctx, 0));
    
    Builder.CreateCall(RegisterKernelParams, {KernelNameStr, ParamCount, NamesPtr, TypesPtr});
  }
  
  Builder.CreateRetVoid();
  
  // 将构造函数加入全局构造函数列表
  appendToGlobalCtors(M, InitFunc, 0);
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
    
    return "unknown_kernel";
  };
  
  return extractRecursive(KernelFunc, 0);
}

std::string CudaArgsProfileImpl::demangleKernelName(const std::string &mangledName) {
  std::string name = mangledName;
  
  // 通用的CUDA/GPU相关前缀清理
  std::vector<std::string> prefixes_to_remove = {
    "__device_stub_", "__global_stub_", "__cuda_", "__hip_",
    "_kernel_stub_", "__kernel_", "kernel_stub_"
  };
  
  for (const auto &prefix : prefixes_to_remove) {
    if (name.find(prefix) == 0) {
      name = name.substr(prefix.length());
      break;
    }
  }
  
  // 通用的C++符号处理
  if (name.find("_Z") == 0) {
    // 简单的C++ demangle
    size_t start = 2;
    if (start < name.length() && std::isdigit(name[start])) {
      size_t len_start = start;
      while (start < name.length() && std::isdigit(name[start])) start++;
      if (len_start < start) {
        // 避免异常处理，使用更安全的方式
        std::string len_str = name.substr(len_start, start - len_start);
        bool valid_num = true;
        for (char c : len_str) {
          if (!std::isdigit(c)) {
            valid_num = false;
            break;
          }
        }
        if (valid_num && !len_str.empty()) {
          int func_len = std::atoi(len_str.c_str());
          if (func_len > 0 && start + func_len <= name.length()) {
            name = name.substr(start, func_len);
          }
        }
      }
    }
  }
  
  // 移除通用的后缀
  std::vector<std::string> suffixes_to_remove = {
    "_stub", "_wrapper", "_impl", "_kernel"
  };
  
  for (const auto &suffix : suffixes_to_remove) {
    size_t pos = name.rfind(suffix);
    if (pos != std::string::npos && pos + suffix.length() == name.length()) {
      name = name.substr(0, pos);
      break;
    }
  }
  
  return name.empty() ? mangledName : name;
}

void CudaArgsProfileImpl::profileDimensions(CallInst *CI, Module &M, 
                                           Value *GridDim, Value *BlockDim) {
  IRBuilder<> Builder(CI);
  LLVMContext &Ctx = M.getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  
  // 默认值
  Value *GridX = ConstantInt::get(Int32Ty, 1);
  Value *GridY = ConstantInt::get(Int32Ty, 1);
  Value *GridZ = ConstantInt::get(Int32Ty, 1);
  
  Value *BlockX = ConstantInt::get(Int32Ty, 1);
  Value *BlockY = ConstantInt::get(Int32Ty, 1);
  Value *BlockZ = ConstantInt::get(Int32Ty, 1);
  
  // 尝试提取实际值
  if (GridDim->getType()->isIntegerTy()) {
    if (GridDim->getType()->getIntegerBitWidth() <= 32) {
      GridX = GridDim;
    } else {
      GridX = Builder.CreateTrunc(GridDim, Int32Ty);
    }
  }
  
  if (BlockDim->getType()->isIntegerTy()) {
    if (BlockDim->getType()->getIntegerBitWidth() <= 32) {
      BlockX = BlockDim;
    } else {
      BlockX = Builder.CreateTrunc(BlockDim, Int32Ty);
    }
  }
  
  Builder.CreateCall(ProfileGridDim, {GridX, GridY, GridZ});
  Builder.CreateCall(ProfileBlockDim, {BlockX, BlockY, BlockZ});
}

void CudaArgsProfileImpl::profileKernelArguments(CallInst *CI, Module &M, Value *ArgsArray, const std::string &kernelName) {
  IRBuilder<> Builder(CI);
  LLVMContext &Ctx = M.getContext();
  Type *Int8PtrTy = PointerType::get(Ctx, 0);
  Type *Int8PtrPtrTy = PointerType::get(Ctx, 0);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  
  Value *ArgsPtrPtr = Builder.CreateBitCast(ArgsArray, Int8PtrPtrTy);
  
  // 查找kernel函数
  Value *KernelFunc = CI->getArgOperand(0);
  Function *KernelFunction = nullptr;
  
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
    
    return nullptr;
  };
  
  KernelFunction = findKernelFunction(KernelFunc, 0);
  
  // 获取参数信息
  std::vector<ParamInfo> paramInfos = getKernelParamInfo(kernelName, KernelFunction);
  
  int numArgs = paramInfos.empty() ? 8 : std::min(static_cast<int>(paramInfos.size()), 16);
  
  for (int i = 0; i < numArgs; i++) {
    Value *ArgIndex = ConstantInt::get(Int32Ty, i);
    Value *ArgPtr = Builder.CreateLoad(Int8PtrTy, 
                                      Builder.CreateGEP(Int8PtrTy, ArgsPtrPtr, 
                                                       ArgIndex));
    
    std::string argName = "param_" + std::to_string(i);
    std::string argType = "unknown";
    int typeHint = 32;
    bool isScalar = true;
    
    if (i < static_cast<int>(paramInfos.size())) {
      argName = paramInfos[i].name;
      argType = paramInfos[i].type;
      typeHint = paramInfos[i].typeHint;
      isScalar = paramInfos[i].isScalar;
    }
    
    // 只profile标量参数
    if (isScalar) {
      Value *ArgNameStr = Builder.CreateGlobalString(argName);
      Value *ArgTypeStr = Builder.CreateGlobalString(argType);
      Value *ArgIndex = ConstantInt::get(Int32Ty, i);
      Value *TypeInfo = ConstantInt::get(Int32Ty, typeHint);
      
      Builder.CreateCall(ProfileScalarArg, {ArgNameStr, ArgIndex, ArgPtr, TypeInfo, ArgTypeStr});
    }
  }
}

bool CudaArgsProfileImpl::instrumentCudaLaunchKernel(CallInst *CI, Module &M) {
  if (CI->getNumOperands() < 8) return false;
  
  IRBuilder<> Builder(CI);
  
  // Extract kernel name and profile kernel launch
  Value *KernelFunc = CI->getArgOperand(0);
  std::string KernelName = extractKernelName(KernelFunc);
  Value *KernelNameStr = Builder.CreateGlobalString(KernelName);
  Builder.CreateCall(ProfileKernelLaunch, {KernelNameStr});
  
  // Profile dimensions
  Value *GridDimXY = CI->getArgOperand(1);
  Value *BlockDimXY = CI->getArgOperand(3);
  profileDimensions(CI, M, GridDimXY, BlockDimXY);
  
  // Profile kernel arguments
  Value *ArgsArray = CI->getArgOperand(5);
  profileKernelArguments(CI, M, ArgsArray, KernelName);
  
  ++CudaKernelLaunches;
  return true;
}

bool CudaArgsProfileImpl::instrumentCudaLaunchKernel_ptsz(CallInst *CI, Module &M) {
  return instrumentCudaLaunchKernel(CI, M);
}

bool CudaArgsProfileImpl::runOnModule(Module &M) {
  bool Modified = false;
  
  createProfileFunctions(M);
  
  // 提前提取调试信息
  extractDebugParamInfo(M);
  
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
  
  // 注册kernel参数信息到runtime
  if (Modified && !KernelParamCache.empty()) {
    registerKernelParametersAtRuntime(M);
  }
  
  return Modified;
}

} // anonymous namespace

PreservedAnalyses CudaArgsProfilePass::run(Module &M, ModuleAnalysisManager &AM) {
  CudaArgsProfileImpl Impl;
  bool Modified = Impl.runOnModule(M);
  return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
} 