//===-- CudaKernelNoalias.cpp - CUDA Kernel Noalias Optimization --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass clones CUDA kernel functions to create noalias versions for
// different pointer parameter combinations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/CudaKernelNoalias/CudaKernelNoalias.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <fstream>
#include <regex>
#include <set>
#include <string>
#include <map>

using namespace llvm;

#define DEBUG_TYPE "cuda-kernel-noalias"

// 命令行选项
static cl::opt<std::string> CudaProfilePath(
    "cuda-kernel-profile",
    cl::desc("Path to the CUDA kernel profile log file"),
    cl::value_desc("filename"),
    cl::init(""));

// 判断是否为kernel函数
static bool isKernelFunction(const Function &F) {
  return F.getCallingConv() == CallingConv::PTX_Kernel;
}

// 判断是否有NVVM kernel注解
static bool hasNVVMKernelAnnotation(const Function &F) {
  const Module *M = F.getParent();
  NamedMDNode *NMD = M->getNamedMetadata("nvvm.annotations");
  if (!NMD)
    return false;

  for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i) {
    const MDNode *MD = NMD->getOperand(i);
    if (MD->getNumOperands() < 3)
      continue;

    if (auto *FMD = mdconst::dyn_extract_or_null<Function>(MD->getOperand(0))) {
      if (FMD == &F) {
        if (auto *Str = dyn_cast<MDString>(MD->getOperand(1))) {
          if (Str->getString() == "kernel")
            return true;
        }
      }
    }
  }
  return false;
}

// 解析profile日志文件
bool CudaKernelNoaliasPass::parseProfileLog() {
  if (CudaProfilePath.empty()) {
    errs() << "Warning: No profile log file specified. Use -mllvm -cuda-kernel-profile=<file>\n";
    return false;
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr = MemoryBuffer::getFile(CudaProfilePath);
  if (!BufferOrErr) {
    errs() << "Error: Cannot open profile log file: " << CudaProfilePath << "\n";
    return false;
  }

  StringRef Content = BufferOrErr.get()->getBuffer();
  Expected<json::Value> Parsed = json::parse(Content);
  if (!Parsed) {
    errs() << "Error: Failed to parse JSON profile log\n";
    return false;
  }

  json::Object *RootObj = Parsed->getAsObject();
  if (!RootObj) {
    errs() << "Error: JSON root is not an object\n";
    return false;
  }

  json::Array *HotKernels = RootObj->getArray("hot_kernels");
  if (!HotKernels) {
    errs() << "Warning: No hot_kernels array in profile log\n";
    return true; // nothing to optimize
  }

  for (json::Value &HKVal : *HotKernels) {
    json::Object *HKObj = HKVal.getAsObject();
    if (!HKObj)
      continue;

    auto NameVal = HKObj->getString("name");
    if (!NameVal)
      continue;

    KernelProfile KP;
    KP.name = *NameVal;

    json::Array *NoaliasArr = HKObj->getArray("noalias_pointers");
    if (NoaliasArr) {
      for (json::Value &PtrVal : *NoaliasArr) {
        json::Object *PtrObj = PtrVal.getAsObject();
        if (!PtrObj)
          continue;
        auto ArgVal = PtrObj->get("arg");
        if (!ArgVal)
          continue;
        if (auto ArgInt = ArgVal->getAsInteger()) {
          PointerInfo PI;
          PI.index = static_cast<unsigned>(*ArgInt);
          KP.pointerParams.push_back(PI);
        }
      }
    }

    if (!KP.pointerParams.empty())
      KernelProfiles[KP.name] = std::move(KP);
  }

  return true;
}

// 获取可以添加noalias属性的指针参数索引
std::vector<unsigned> CudaKernelNoaliasPass::getNoAliasPointerIndices(const Function &F) {
  std::vector<unsigned> NoAliasParams;
  auto It = KernelProfiles.find(F.getName().str());
  
  if (It != KernelProfiles.end()) {
    for (const auto &PI : It->second.pointerParams) {
      NoAliasParams.push_back(PI.index);
    }
  }
  
  return NoAliasParams;
}

static Function *cloneKernelWithNoalias(Function &OrigF, 
                                      const std::vector<unsigned> &NoaliasParams,
                                      const std::string &Suffix) {
  ValueToValueMapTy VMap;
  Function *ClonedF = CloneFunction(&OrigF, VMap);
  
  // 设置新名称
  std::string NewName = OrigF.getName().str() + "_" + Suffix;
  ClonedF->setName(NewName);
  
  // 为指定的参数添加noalias属性
  for (unsigned Idx : NoaliasParams) {
    if (Idx < ClonedF->arg_size()) {
      ClonedF->addParamAttr(Idx, Attribute::NoAlias);
    }
  }
  
  // 复制nvvm.annotations元数据
  Module *M = OrigF.getParent();
  NamedMDNode *NMD = M->getOrInsertNamedMetadata("nvvm.annotations");
  
  for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i) {
    const MDNode *MD = NMD->getOperand(i);
    if (MD->getNumOperands() >= 3) {
      if (auto *FMD = mdconst::dyn_extract_or_null<Function>(MD->getOperand(0))) {
        if (FMD == &OrigF) {
          if (auto *Str = dyn_cast<MDString>(MD->getOperand(1))) {
            if (Str->getString() == "kernel") {
              LLVMContext &Ctx = M->getContext();
              Metadata *MDVals[] = {
                ValueAsMetadata::get(ClonedF),
                MDString::get(Ctx, "kernel"),
                MD->getOperand(2)
              };
              MDNode *NewMD = MDNode::get(Ctx, MDVals);
              NMD->addOperand(NewMD);
              break;
            }
          }
        }
      }
    }
  }
  
  return ClonedF;
}

PreservedAnalyses CudaKernelNoaliasPass::run(Module &M, ModuleAnalysisManager &AM) {
  if (!parseProfileLog()) {
    errs() << "Warning: Failed to parse profile log, skipping noalias optimization\n";
    return PreservedAnalyses::all();
  }

  bool Changed = false;
  std::vector<Function *> KernelsToProcess;
  
  // 识别所有kernel函数
  for (Function &F : M) {
    if (isKernelFunction(F)) {
      KernelsToProcess.push_back(&F);
    }
  }
  
  // 处理每个kernel函数
  for (Function *F : KernelsToProcess) {
    std::vector<unsigned> NoAliasParams = getNoAliasPointerIndices(*F);
    
    // 如果有可以添加noalias的参数，创建优化版本
    if (!NoAliasParams.empty()) {
      std::string Suffix = "noalias";
      Function *ClonedF = cloneKernelWithNoalias(*F, NoAliasParams, Suffix);
      Changed = true;
      
      LLVM_DEBUG(dbgs() << "Cloned kernel " << F->getName() 
                       << " to " << ClonedF->getName() 
                       << " with noalias on " << NoAliasParams.size() 
                       << " pointer parameters\n");
    }
  }
  
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
} 