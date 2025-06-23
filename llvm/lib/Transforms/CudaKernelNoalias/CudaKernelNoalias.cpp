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
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <algorithm>
#include <set>
#include <string>

using namespace llvm;

#define DEBUG_TYPE "cuda-kernel-noalias"

static bool isKernelFunction(const Function &F) {
  return F.getCallingConv() == CallingConv::PTX_Kernel;
}

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

static std::vector<unsigned> getPointerParameterIndices(const Function &F) {
  std::vector<unsigned> PointerParams;
  for (const auto &Arg : F.args()) {
    if (Arg.getType()->isPointerTy()) {
      PointerParams.push_back(Arg.getArgNo());
    }
  }
  return PointerParams;
}

static Function *cloneKernelWithNoalias(Function &OrigF, 
                                        const std::vector<unsigned> &NoaliasParams,
                                        const std::string &Suffix) {
  ValueToValueMapTy VMap;
  Function *ClonedF = CloneFunction(&OrigF, VMap);
  
  // Set the new name
  std::string NewName = OrigF.getName().str() + "_" + Suffix;
  ClonedF->setName(NewName);
  
  // Add noalias attributes to the specified parameters
  for (unsigned Idx : NoaliasParams) {
    if (Idx < ClonedF->arg_size()) {
      ClonedF->addParamAttr(Idx, Attribute::NoAlias);
    }
  }
  
  // Copy nvvm.annotations metadata for the cloned function
  Module *M = OrigF.getParent();
  NamedMDNode *NMD = M->getOrInsertNamedMetadata("nvvm.annotations");
  
  // Find existing kernel annotation and clone it
  for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i) {
    const MDNode *MD = NMD->getOperand(i);
    if (MD->getNumOperands() >= 3) {
      if (auto *FMD = mdconst::dyn_extract_or_null<Function>(MD->getOperand(0))) {
        if (FMD == &OrigF) {
          if (auto *Str = dyn_cast<MDString>(MD->getOperand(1))) {
            if (Str->getString() == "kernel") {
              // Create new metadata for cloned function
              LLVMContext &Ctx = M->getContext();
              Metadata *MDVals[] = {
                ValueAsMetadata::get(ClonedF),
                MDString::get(Ctx, "kernel"),
                MD->getOperand(2) // kernel flag value
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

static std::string generateSuffix(const std::vector<unsigned> &Indices,
                                  const Function &F) {
  std::string Suffix;
  for (size_t i = 0; i < Indices.size(); ++i) {
    if (i > 0) Suffix += "_";
    // Use parameter names if available, otherwise use indices
    if (Indices[i] < F.arg_size()) {
      const Argument *Arg = F.getArg(Indices[i]);
      if (Arg->hasName()) {
        Suffix += Arg->getName().str();
      } else {
        Suffix += "arg" + std::to_string(Indices[i]);
      }
    }
  }
  return Suffix;
}

PreservedAnalyses CudaKernelNoaliasPass::run(Module &M, ModuleAnalysisManager &AM) {
  bool Changed = false;
  std::vector<Function *> KernelsToProcess;
  
  // First, identify all kernel functions
  for (Function &F : M) {
    if (isKernelFunction(F)) {
      KernelsToProcess.push_back(&F);
    }
  }
  
  // Process each kernel function
  for (Function *F : KernelsToProcess) {
    std::vector<unsigned> PointerParams = getPointerParameterIndices(*F);
    
    // Skip if less than 2 pointer parameters
    if (PointerParams.size() < 2) continue;
    
    // Create a single noalias version with all pointer parameters marked as noalias
    std::string Suffix = "noalias";
    Function *ClonedF = cloneKernelWithNoalias(*F, PointerParams, Suffix);
    Changed = true;
    
    LLVM_DEBUG(dbgs() << "Cloned kernel " << F->getName() 
                     << " to " << ClonedF->getName() 
                     << " with noalias on all " << PointerParams.size() 
                     << " pointer parameters\n");
  }
  
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
} 