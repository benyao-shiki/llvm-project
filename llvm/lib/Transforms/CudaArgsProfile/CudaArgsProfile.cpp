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
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include <sstream>
#include <vector>
#include <string>

using namespace llvm;

namespace {

struct ParamTypeInfo {
  enum Type {
    SCALAR_INT8, SCALAR_INT16, SCALAR_INT32, SCALAR_INT64,
    SCALAR_FLOAT, SCALAR_DOUBLE,
    POINTER,
    STRUCT
  };
  
  Type type;
  std::string name;
  size_t size;
  size_t offset;
  std::vector<ParamTypeInfo> members; // For struct types
};

// 新增：参数映射信息结构
struct ArgMappingInfo {
  std::string originalName;           // 原始参数名
  ParamTypeInfo originalParam;        // 原始参数信息
  std::vector<unsigned> argsIndices;  // 在args数组中的索引
  std::vector<Type*> splitTypes;      // 拆分后的IR类型
  bool isSplit;                       // 是否被拆分
  
  ArgMappingInfo() : isSplit(false) {}
};

struct KernelInfo {
  std::string name;
  std::vector<ParamTypeInfo> params;
  std::vector<ArgMappingInfo> argMappings;  // 新增：参数映射信息
  size_t totalParamSize;
};

class CudaArgsProfileImpl {
private:
  Module *M;
  LLVMContext *Context;
  
  // Runtime functions
  Function *ProfileKernelFunc;
  Function *ProfileStartFunc;
  Function *ProfileEndFunc;
  Function *ProfileScalarFunc;
  Function *ProfileStructFunc;
  Function *ProfileStructValueFunc;
  Function *ProfilePointerFunc;
  Function *ProfileSplitStructFunc;  // 新增：处理拆分结构体的函数
  
  // Global variables
  GlobalVariable *KernelCounterGV;
  GlobalVariable *OutputFileGV;
  
  std::vector<KernelInfo> KernelInfos;
  
public:
  CudaArgsProfileImpl(Module &Mod) : M(&Mod), Context(&Mod.getContext()) {
    createRuntimeFunctions();
    createGlobalVariables();
  }
  
  bool runOnModule();
  
private:
  void createRuntimeFunctions();
  void createGlobalVariables();
  bool instrumentCudaLaunchCalls();
  bool isCudaLaunchCall(CallInst *CI);
  Function *getKernelFunction(Value *KernelArg);
  KernelInfo analyzeKernelFunction(Function *KernelFunc);
  std::vector<ArgMappingInfo> analyzeStructSplitting(Function *KernelFunc);  // 新增
  ArgMappingInfo analyzeParameterDebugInfo(DIType *ParamType, const std::vector<Type*> &IRArgTypes, unsigned &currentIRArgIndex);  // 新增
  ParamTypeInfo analyzeType(Type *Ty, const std::string &Name = "");
  ParamTypeInfo analyzeDebugType(DIType *DebugType);  // 新增
  void insertProfilingCalls(CallInst *LaunchCall, const KernelInfo &KInfo);
  void handleSplitStructArgument(IRBuilder<> &Builder, Value *Args, const ArgMappingInfo &mapping, unsigned paramIndex);  // 新增
  std::string generateKernelInfoString(const KernelInfo &KInfo);
  std::string generateStructMemberInfo(const ParamTypeInfo &StructInfo);
  void createModuleInitializer();
  void createModuleFinalizer();
};

void CudaArgsProfileImpl::createRuntimeFunctions() {
  // void __cuda_profile_kernel_start(const char* kernel_name, int kernel_id, 
  //                                  int gridx, int gridy, int gridz,
  //                                  int blockx, int blocky, int blockz)
  Type *VoidTy = Type::getVoidTy(*Context);
  Type *Int32Ty = Type::getInt32Ty(*Context);
  Type *CharPtrTy = PointerType::getUnqual(Type::getInt8Ty(*Context));
  
  ProfileStartFunc = Function::Create(
    FunctionType::get(VoidTy, {CharPtrTy, Int32Ty, Int32Ty, Int32Ty, Int32Ty, 
                               Int32Ty, Int32Ty, Int32Ty}, false),
    Function::ExternalLinkage, "__cuda_profile_kernel_start", M);
  
  // void __cuda_profile_kernel_end()
  ProfileEndFunc = Function::Create(
    FunctionType::get(VoidTy, {}, false),
    Function::ExternalLinkage, "__cuda_profile_kernel_end", M);
  
  // void __cuda_profile_scalar(int index, int type, void* value_ptr, size_t size)
  Type *SizeTy = Type::getInt64Ty(*Context);
  Type *VoidPtrTy = PointerType::getUnqual(Type::getInt8Ty(*Context));
  ProfileScalarFunc = Function::Create(
    FunctionType::get(VoidTy, {Int32Ty, Int32Ty, VoidPtrTy, SizeTy}, false),
    Function::ExternalLinkage, "__cuda_profile_scalar", M);
  
  // void __cuda_profile_struct_start(int index, const char* struct_name, size_t size)
  ProfileStructFunc = Function::Create(
    FunctionType::get(VoidTy, {Int32Ty, CharPtrTy, SizeTy}, false),
    Function::ExternalLinkage, "__cuda_profile_struct_start", M);
  
  // void __cuda_profile_struct_value(int index, const char* struct_name, void* value_ptr, size_t size, const char* member_info)
  ProfileStructValueFunc = Function::Create(
    FunctionType::get(VoidTy, {Int32Ty, CharPtrTy, VoidPtrTy, SizeTy, CharPtrTy}, false),
    Function::ExternalLinkage, "__cuda_profile_struct_value", M);
  
  // void __cuda_profile_pointer(int index, void* ptr_value)
  ProfilePointerFunc = Function::Create(
    FunctionType::get(VoidTy, {Int32Ty, VoidPtrTy}, false),
    Function::ExternalLinkage, "__cuda_profile_pointer", M);
  
  // 新增：void __cuda_profile_split_struct(int index, const char* struct_name, 
  //                                        void** split_parts, int num_parts, 
  //                                        const char* member_info)
  Type *VoidPtrPtrTy = PointerType::getUnqual(VoidPtrTy);
  ProfileSplitStructFunc = Function::Create(
    FunctionType::get(VoidTy, {Int32Ty, CharPtrTy, VoidPtrPtrTy, Int32Ty, CharPtrTy}, false),
    Function::ExternalLinkage, "__cuda_profile_split_struct", M);
}

void CudaArgsProfileImpl::createGlobalVariables() {
  // Global counter for kernel IDs
  KernelCounterGV = new GlobalVariable(
    *M, Type::getInt32Ty(*Context), false, GlobalValue::InternalLinkage,
    ConstantInt::get(Type::getInt32Ty(*Context), 0), "__cuda_kernel_counter");
  
  // Output file path
  PointerType *CharPtrTy = PointerType::getUnqual(Type::getInt8Ty(*Context));
  OutputFileGV = new GlobalVariable(
    *M, CharPtrTy, false, GlobalValue::InternalLinkage,
    ConstantPointerNull::get(CharPtrTy), "__cuda_output_file");
}

bool CudaArgsProfileImpl::runOnModule() {
  bool Changed = false;
  
  // Instrument CUDA launch calls
  Changed |= instrumentCudaLaunchCalls();
  
  // Create module initializer and finalizer
  if (Changed) {
    createModuleInitializer();
    createModuleFinalizer();
  }
  
  return Changed;
}

bool CudaArgsProfileImpl::instrumentCudaLaunchCalls() {
  bool Changed = false;
  
  for (Function &F : *M) {
    for (BasicBlock &BB : F) {
      for (auto I = BB.begin(), E = BB.end(); I != E; ++I) {
        if (CallInst *CI = dyn_cast<CallInst>(&*I)) {
          if (isCudaLaunchCall(CI)) {
            Function *KernelFunc = getKernelFunction(CI->getArgOperand(0));
            if (KernelFunc) {
              KernelInfo KInfo = analyzeKernelFunction(KernelFunc);
              insertProfilingCalls(CI, KInfo);
              Changed = true;
            }
          }
        }
      }
    }
  }
  
  return Changed;
}

bool CudaArgsProfileImpl::isCudaLaunchCall(CallInst *CI) {
  Function *Callee = CI->getCalledFunction();
  if (!Callee) return false;
  
  StringRef Name = Callee->getName();
  return Name == "cudaLaunchKernel" || Name == "cudaLaunchKernelExMem" ||
         Name == "__cudaLaunchKernel" || Name == "__cudaLaunchKernelExMem";
}

Function *CudaArgsProfileImpl::getKernelFunction(Value *KernelArg) {
  // Handle bitcasts and other casts
  if (BitCastInst *BCI = dyn_cast<BitCastInst>(KernelArg)) {
    KernelArg = BCI->getOperand(0);
  }
  
  // Direct function reference
  if (Function *F = dyn_cast<Function>(KernelArg)) {
    return F;
  }
  
  // Global variable (function pointer)
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(KernelArg)) {
    if (GV->hasInitializer()) {
      if (Function *F = dyn_cast<Function>(GV->getInitializer())) {
        return F;
      }
    }
  }
  
  return nullptr;
}

KernelInfo CudaArgsProfileImpl::analyzeKernelFunction(Function *KernelFunc) {
  KernelInfo KInfo;
  KInfo.name = KernelFunc->getName().str();
  KInfo.totalParamSize = 0;
  
  // 首先尝试使用调试信息分析结构体拆分
  KInfo.argMappings = analyzeStructSplitting(KernelFunc);
  
  if (!KInfo.argMappings.empty()) {
    // 使用调试信息分析的结果
    for (const auto &mapping : KInfo.argMappings) {
      KInfo.params.push_back(mapping.originalParam);
      KInfo.totalParamSize += mapping.originalParam.size;
    }
  } else {
    // 降级到原始的类型分析
    for (auto &Arg : KernelFunc->args()) {
      ParamTypeInfo ParamInfo = analyzeType(Arg.getType(), Arg.getName().str());
      KInfo.params.push_back(ParamInfo);
      KInfo.totalParamSize += ParamInfo.size;
    }
  }
  
  return KInfo;
}

// 新增：分析结构体拆分的核心函数
std::vector<ArgMappingInfo> CudaArgsProfileImpl::analyzeStructSplitting(Function *KernelFunc) {
  std::vector<ArgMappingInfo> mappings;
  
  // 获取函数的调试信息
  DISubprogram *SP = KernelFunc->getSubprogram();
  if (!SP) {
    return mappings;  // 没有调试信息，返回空
  }
  
  // 获取函数类型的调试信息
  DISubroutineType *FuncType = SP->getType();
  if (!FuncType) return mappings;
  
  DITypeRefArray TypeArray = FuncType->getTypeArray();
  if (!TypeArray) return mappings;
  
  // 获取IR参数类型列表
  std::vector<Type*> IRArgTypes;
  for (auto &Arg : KernelFunc->args()) {
    IRArgTypes.push_back(Arg.getType());
  }
  
  // 分析每个原始参数（跳过返回类型）
  unsigned currentIRArgIndex = 0;
  for (unsigned i = 1; i < TypeArray.size() && currentIRArgIndex < IRArgTypes.size(); ++i) {
    if (DIType *ParamType = TypeArray[i]) {
      ArgMappingInfo mapping = analyzeParameterDebugInfo(ParamType, IRArgTypes, currentIRArgIndex);
      if (!mapping.originalName.empty()) {
        mappings.push_back(mapping);
      }
    }
  }
  
  return mappings;
}

// 新增：分析单个参数的调试信息
ArgMappingInfo CudaArgsProfileImpl::analyzeParameterDebugInfo(DIType *ParamType, 
                                                             const std::vector<Type*> &IRArgTypes, 
                                                             unsigned &currentIRArgIndex) {
  ArgMappingInfo mapping;
  
  if (currentIRArgIndex >= IRArgTypes.size()) {
    return mapping;  // 防止越界
  }
  
  // 如果是结构体类型
  if (DICompositeType *CompositeType = dyn_cast<DICompositeType>(ParamType)) {
    if (CompositeType->getTag() == dwarf::DW_TAG_structure_type) {
      mapping.originalName = CompositeType->getName().str();
      mapping.originalParam = analyzeDebugType(CompositeType);
      
      uint64_t structSizeBits = CompositeType->getSizeInBits();
      
      // 根据结构体大小判断拆分模式
      if (structSizeBits <= 64) {
        // 8字节或更小 -> 单个参数
        mapping.isSplit = false;
        mapping.argsIndices.push_back(currentIRArgIndex);
        mapping.splitTypes.push_back(IRArgTypes[currentIRArgIndex]);
        currentIRArgIndex++;
      } else if (structSizeBits <= 128) {
        // 16字节 -> 通常拆分为两个8字节参数
        mapping.isSplit = true;
        mapping.argsIndices.push_back(currentIRArgIndex);
        mapping.argsIndices.push_back(currentIRArgIndex + 1);
        mapping.splitTypes.push_back(IRArgTypes[currentIRArgIndex]);
        mapping.splitTypes.push_back(IRArgTypes[currentIRArgIndex + 1]);
        currentIRArgIndex += 2;
      } else {
        // 更大的结构体 -> byval传递
        mapping.isSplit = false;
        mapping.argsIndices.push_back(currentIRArgIndex);
        mapping.splitTypes.push_back(IRArgTypes[currentIRArgIndex]);
        currentIRArgIndex++;
      }
      
      return mapping;
    }
  }
  
  // 非结构体类型
  mapping.originalParam = analyzeType(IRArgTypes[currentIRArgIndex]);
  mapping.isSplit = false;
  mapping.argsIndices.push_back(currentIRArgIndex);
  mapping.splitTypes.push_back(IRArgTypes[currentIRArgIndex]);
  currentIRArgIndex++;
  
  return mapping;
}

// 新增：从调试信息分析类型
ParamTypeInfo CudaArgsProfileImpl::analyzeDebugType(DIType *DebugType) {
  ParamTypeInfo info;
  
  if (DICompositeType *CompositeType = dyn_cast<DICompositeType>(DebugType)) {
    if (CompositeType->getTag() == dwarf::DW_TAG_structure_type) {
      info.type = ParamTypeInfo::STRUCT;
      info.name = CompositeType->getName().str();
      info.size = CompositeType->getSizeInBits() / 8;  // 转换为字节
      
      // 分析结构体成员
      DINodeArray Elements = CompositeType->getElements();
      for (DINode *Element : Elements) {
        if (DIDerivedType *Member = dyn_cast<DIDerivedType>(Element)) {
          if (Member->getTag() == dwarf::DW_TAG_member) {
            ParamTypeInfo memberInfo;
            memberInfo.name = Member->getName().str();
            memberInfo.offset = Member->getOffsetInBits() / 8;  // 转换为字节
            memberInfo.size = Member->getSizeInBits() / 8;
            
            // 根据基础类型设置成员类型
            if (DIType *BaseType = Member->getBaseType()) {
              if (DIBasicType *BasicType = dyn_cast<DIBasicType>(BaseType)) {
                unsigned encoding = BasicType->getEncoding();
                unsigned sizeBits = BasicType->getSizeInBits();
                
                if (encoding == dwarf::DW_ATE_signed || encoding == dwarf::DW_ATE_unsigned) {
                  switch (sizeBits) {
                    case 8:  memberInfo.type = ParamTypeInfo::SCALAR_INT8; break;
                    case 16: memberInfo.type = ParamTypeInfo::SCALAR_INT16; break;
                    case 32: memberInfo.type = ParamTypeInfo::SCALAR_INT32; break;
                    case 64: memberInfo.type = ParamTypeInfo::SCALAR_INT64; break;
                    default: memberInfo.type = ParamTypeInfo::SCALAR_INT32; break;
                  }
                } else if (encoding == dwarf::DW_ATE_float) {
                  if (sizeBits == 32) {
                    memberInfo.type = ParamTypeInfo::SCALAR_FLOAT;
                  } else if (sizeBits == 64) {
                    memberInfo.type = ParamTypeInfo::SCALAR_DOUBLE;
                  }
                }
              }
            }
            
            info.members.push_back(memberInfo);
          }
        }
      }
    }
  }
  
  return info;
}

ParamTypeInfo CudaArgsProfileImpl::analyzeType(Type *Ty, const std::string &Name) {
  ParamTypeInfo Info;
  Info.name = Name;
  
  if (Ty->isPointerTy()) {
    Info.type = ParamTypeInfo::POINTER;
    Info.size = M->getDataLayout().getPointerSize();
  } else if (Ty->isIntegerTy()) {
    unsigned BitWidth = Ty->getIntegerBitWidth();
    Info.size = BitWidth / 8;
    switch (BitWidth) {
      case 8:  Info.type = ParamTypeInfo::SCALAR_INT8; break;
      case 16: Info.type = ParamTypeInfo::SCALAR_INT16; break;
      case 32: Info.type = ParamTypeInfo::SCALAR_INT32; break;
      case 64: Info.type = ParamTypeInfo::SCALAR_INT64; break;
      default: Info.type = ParamTypeInfo::SCALAR_INT32; break;
    }
  } else if (Ty->isFloatTy()) {
    Info.type = ParamTypeInfo::SCALAR_FLOAT;
    Info.size = 4;
  } else if (Ty->isDoubleTy()) {
    Info.type = ParamTypeInfo::SCALAR_DOUBLE;
    Info.size = 8;
  } else if (StructType *STy = dyn_cast<StructType>(Ty)) {
    Info.type = ParamTypeInfo::STRUCT;
    Info.size = M->getDataLayout().getTypeAllocSize(STy);
    
    // Analyze struct members (one level deep)
    const StructLayout *SL = M->getDataLayout().getStructLayout(STy);
    for (unsigned i = 0; i < STy->getNumElements(); ++i) {
      Type *MemberTy = STy->getElementType(i);
      std::string MemberName = "member_" + std::to_string(i);
      ParamTypeInfo MemberInfo = analyzeType(MemberTy, MemberName);
      MemberInfo.offset = SL->getElementOffset(i);
      Info.members.push_back(MemberInfo);
    }
  } else {
    // Unknown type, treat as raw bytes
    Info.type = ParamTypeInfo::SCALAR_INT32;
    Info.size = M->getDataLayout().getTypeAllocSize(Ty);
  }
  
  return Info;
}

void CudaArgsProfileImpl::insertProfilingCalls(CallInst *LaunchCall, 
                                               const KernelInfo &KInfo) {
  IRBuilder<> Builder(LaunchCall);
  
  // Get launch parameters
  Value *Args = LaunchCall->getArgOperand(5);
  
  // Load and increment kernel counter
  Value *KernelId = Builder.CreateLoad(Type::getInt32Ty(*Context), KernelCounterGV);
  Value *NewId = Builder.CreateAdd(KernelId, Builder.getInt32(1));
  Builder.CreateStore(NewId, KernelCounterGV);
  
  // For now, use default grid and block dimensions
  Value *GridX = Builder.getInt32(1);
  Value *GridY = Builder.getInt32(1);
  Value *GridZ = Builder.getInt32(1);
  Value *BlockX = Builder.getInt32(1);
  Value *BlockY = Builder.getInt32(1);
  Value *BlockZ = Builder.getInt32(1);
  
  // Create kernel name string
  std::string KernelInfoStr = generateKernelInfoString(KInfo);
  Value *KernelNameStr = Builder.CreateGlobalString(KernelInfoStr);
  Type *CharPtrTy = PointerType::getUnqual(Type::getInt8Ty(*Context));
  Value *KernelNamePtr = Builder.CreatePointerCast(KernelNameStr, CharPtrTy);
  
  // Call profile start
  Builder.CreateCall(ProfileStartFunc, {KernelNamePtr, KernelId, GridX, GridY, GridZ, 
                                        BlockX, BlockY, BlockZ});
  
  // Profile each argument using mapping information
  Type *VoidPtrTy = PointerType::getUnqual(Type::getInt8Ty(*Context));
  
  if (!KInfo.argMappings.empty()) {
    // 使用调试信息的映射
    for (size_t i = 0; i < KInfo.argMappings.size(); ++i) {
      const ArgMappingInfo &mapping = KInfo.argMappings[i];
      
      if (mapping.isSplit) {
        // 处理拆分的结构体参数
        handleSplitStructArgument(Builder, Args, mapping, i);
      } else {
        // 处理普通参数
        unsigned argIndex = mapping.argsIndices[0];
        Value *ArgPtr = Builder.CreateGEP(VoidPtrTy, Args, Builder.getInt32(argIndex));
        Value *ArgValuePtr = Builder.CreateLoad(VoidPtrTy, ArgPtr);
        
        const ParamTypeInfo &ParamInfo = mapping.originalParam;
        
        if (ParamInfo.type == ParamTypeInfo::POINTER) {
          Value *PointerValue = Builder.CreateLoad(VoidPtrTy, ArgValuePtr);
          Builder.CreateCall(ProfilePointerFunc, {Builder.getInt32(i), PointerValue});
        } else if (ParamInfo.type == ParamTypeInfo::STRUCT) {
          std::string StructName = ParamInfo.name.empty() ? "struct_" + std::to_string(i) : ParamInfo.name;
          Value *StructNameStr = Builder.CreateGlobalString(StructName);
          Value *StructNamePtr = Builder.CreatePointerCast(StructNameStr, CharPtrTy);
          
          std::string MemberInfoStr = generateStructMemberInfo(ParamInfo);
          Value *MemberInfoStrVal = Builder.CreateGlobalString(MemberInfoStr);
          Value *MemberInfoPtr = Builder.CreatePointerCast(MemberInfoStrVal, CharPtrTy);
          
          Builder.CreateCall(ProfileStructValueFunc, {Builder.getInt32(i), StructNamePtr, 
                                                      ArgValuePtr, Builder.getInt64(ParamInfo.size), 
                                                      MemberInfoPtr});
        } else {
          Builder.CreateCall(ProfileScalarFunc, {Builder.getInt32(i), 
                                                 Builder.getInt32(ParamInfo.type),
                                                 ArgValuePtr, Builder.getInt64(ParamInfo.size)});
        }
      }
    }
  } else {
    // 降级到原始处理方式
    for (size_t i = 0; i < KInfo.params.size(); ++i) {
      const ParamTypeInfo &ParamInfo = KInfo.params[i];
      
      Value *ArgPtr = Builder.CreateGEP(VoidPtrTy, Args, Builder.getInt32(i));
      Value *ArgValuePtr = Builder.CreateLoad(VoidPtrTy, ArgPtr);
      
      if (ParamInfo.type == ParamTypeInfo::POINTER) {
        Value *PointerValue = Builder.CreateLoad(VoidPtrTy, ArgValuePtr);
        Builder.CreateCall(ProfilePointerFunc, {Builder.getInt32(i), PointerValue});
      } else if (ParamInfo.type == ParamTypeInfo::STRUCT) {
        std::string StructName = ParamInfo.name.empty() ? "struct_" + std::to_string(i) : ParamInfo.name;
        Value *StructNameStr = Builder.CreateGlobalString(StructName);
        Value *StructNamePtr = Builder.CreatePointerCast(StructNameStr, CharPtrTy);
        
        std::string MemberInfoStr = generateStructMemberInfo(ParamInfo);
        Value *MemberInfoStrVal = Builder.CreateGlobalString(MemberInfoStr);
        Value *MemberInfoPtr = Builder.CreatePointerCast(MemberInfoStrVal, CharPtrTy);
        
        Builder.CreateCall(ProfileStructValueFunc, {Builder.getInt32(i), StructNamePtr, 
                                                    ArgValuePtr, Builder.getInt64(ParamInfo.size), 
                                                    MemberInfoPtr});
      } else {
        Builder.CreateCall(ProfileScalarFunc, {Builder.getInt32(i), 
                                               Builder.getInt32(ParamInfo.type),
                                               ArgValuePtr, Builder.getInt64(ParamInfo.size)});
      }
    }
  }
  
  // Call profile end
  Builder.CreateCall(ProfileEndFunc);
}

// 新增：处理拆分的结构体参数
void CudaArgsProfileImpl::handleSplitStructArgument(IRBuilder<> &Builder, Value *Args, 
                                                   const ArgMappingInfo &mapping, 
                                                   unsigned paramIndex) {
  Type *VoidPtrTy = PointerType::getUnqual(Type::getInt8Ty(*Context));
  Type *CharPtrTy = PointerType::getUnqual(Type::getInt8Ty(*Context));
  
  // 创建结构体名称字符串
  std::string StructName = mapping.originalName.empty() ? 
                          "struct_" + std::to_string(paramIndex) : mapping.originalName;
  Value *StructNameStr = Builder.CreateGlobalString(StructName);
  Value *StructNamePtr = Builder.CreatePointerCast(StructNameStr, CharPtrTy);
  
  // 创建成员信息字符串
  std::string MemberInfoStr = generateStructMemberInfo(mapping.originalParam);
  Value *MemberInfoStrVal = Builder.CreateGlobalString(MemberInfoStr);
  Value *MemberInfoPtr = Builder.CreatePointerCast(MemberInfoStrVal, CharPtrTy);
  
  // 获取完整的结构体数据 - Args[paramIndex]包含完整的结构体
  Value *StructArgPtr = Builder.CreateGEP(VoidPtrTy, Args, Builder.getInt32(paramIndex));
  Value *StructDataPtr = Builder.CreateLoad(VoidPtrTy, StructArgPtr);
  
  // 创建拆分部分的指针数组
  // 我们需要根据结构体的内存布局来创建split parts
  size_t structSize = mapping.originalParam.size;
  size_t numParts = (structSize + 7) / 8;  // 8字节对齐的部分数量
  
  ArrayType *PtrArrayType = ArrayType::get(VoidPtrTy, numParts);
  Value *SplitParts = Builder.CreateAlloca(PtrArrayType);
  
  // 填充拆分部分的指针 - 每个part是8字节对齐的
  for (size_t i = 0; i < numParts; ++i) {
    // 计算偏移量
    Value *OffsetValue = Builder.getInt32(i * 8);
    Value *PartPtr = Builder.CreateGEP(Type::getInt8Ty(*Context), StructDataPtr, OffsetValue);
    
    Value *ArrayElementPtr = Builder.CreateGEP(PtrArrayType, SplitParts, 
                                               {Builder.getInt32(0), Builder.getInt32(i)});
    Builder.CreateStore(PartPtr, ArrayElementPtr);
  }
  
  // 转换为void**
  Value *SplitPartsPtr = Builder.CreatePointerCast(SplitParts, 
                                                    PointerType::getUnqual(VoidPtrTy));
  
  // 调用拆分结构体的profiling函数
  Builder.CreateCall(ProfileSplitStructFunc, {
    Builder.getInt32(paramIndex), 
    StructNamePtr, 
    SplitPartsPtr, 
    Builder.getInt32(numParts),
    MemberInfoPtr
  });
}

std::string CudaArgsProfileImpl::generateKernelInfoString(const KernelInfo &KInfo) {
  std::ostringstream oss;
  oss << KInfo.name << "|" << KInfo.params.size();
  
  for (const ParamTypeInfo &Param : KInfo.params) {
    oss << "|" << Param.type << ":" << Param.size;
    if (Param.type == ParamTypeInfo::STRUCT) {
      oss << ":" << Param.members.size();
      for (const ParamTypeInfo &Member : Param.members) {
        oss << ":" << Member.type << ":" << Member.offset << ":" << Member.size;
      }
    }
  }
  
  return oss.str();
}

std::string CudaArgsProfileImpl::generateStructMemberInfo(const ParamTypeInfo &StructInfo) {
  std::ostringstream oss;
  oss << StructInfo.members.size();
  
  for (const ParamTypeInfo &Member : StructInfo.members) {
    oss << ":" << Member.name << ":" << Member.type << ":" << Member.offset << ":" << Member.size;
  }
  
  return oss.str();
}

void CudaArgsProfileImpl::createModuleInitializer() {
  // Create __cuda_profile_init function
  Function *InitFunc = Function::Create(
    FunctionType::get(Type::getVoidTy(*Context), {}, false),
    Function::InternalLinkage, "__cuda_profile_init", M);
  
  BasicBlock *Entry = BasicBlock::Create(*Context, "entry", InitFunc);
  IRBuilder<> Builder(Entry);
  
  // Initialize output file
  Value *OutputFile = Builder.CreateGlobalString("cuda_profile.json");
  Type *CharPtrTy = PointerType::getUnqual(Type::getInt8Ty(*Context));
  Value *OutputFilePtr = Builder.CreatePointerCast(OutputFile, CharPtrTy);
  Builder.CreateStore(OutputFilePtr, OutputFileGV);
  
  Builder.CreateRetVoid();
  
  // Add to global constructors
  appendToGlobalCtors(*M, InitFunc, 0);
}

void CudaArgsProfileImpl::createModuleFinalizer() {
  // Create __cuda_profile_fini function
  Function *FiniFunc = Function::Create(
    FunctionType::get(Type::getVoidTy(*Context), {}, false),
    Function::InternalLinkage, "__cuda_profile_fini", M);
  
  BasicBlock *Entry = BasicBlock::Create(*Context, "entry", FiniFunc);
  IRBuilder<> Builder(Entry);
  
  // Create external function to finalize profiling
  Function *ProfileFinalizeFunc = Function::Create(
    FunctionType::get(Type::getVoidTy(*Context), {}, false),
    Function::ExternalLinkage, "__cuda_profile_finalize", M);
  
  Builder.CreateCall(ProfileFinalizeFunc);
  Builder.CreateRetVoid();
  
  // Add to global destructors
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