//===-- CudaKernelConst.cpp - CUDA Kernel Const Propagation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass clones CUDA kernel functions to create constant propagation
// versions based on profiling information for common scalar and dimension
// values.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/CudaKernelConst/CudaKernelConst.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
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

#define DEBUG_TYPE "cuda-kernel-const"

// Command-line option for specifying the profile JSON file path
static cl::opt<std::string> CudaConstProfilePath(
    "cuda-kernel-const-profile",
    cl::desc("Path to the CUDA kernel profile log file for constant propagation"),
    cl::value_desc("filename"),
    cl::init(""));

// Return true if the given function is a PTX kernel entry
static bool isKernelFunction(const Function &F) {
  return F.getCallingConv() == CallingConv::PTX_Kernel;
}

// Utility: check whether the function has an explicit NVVM "kernel" annotation
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

// Parse the JSON profile log and populate KernelProfiles
bool CudaKernelConstPass::parseProfileLog() {
  std::string ProfilePath = CudaConstProfilePath;
  if (ProfilePath.empty()) {
    errs() << "Warning: No profile log file specified for constant propagation. "
           << "Use -mllvm -cuda-kernel-const-profile=<file>\n";
    return false;
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr = MemoryBuffer::getFile(ProfilePath);
  if (!BufferOrErr) {
    errs() << "Error: Cannot open profile log file: " << ProfilePath << "\n";
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
    return true;
  }

  for (json::Value &HKVal : *HotKernels) {
    json::Object *HKObj = HKVal.getAsObject();
    if (!HKObj)
      continue;

    auto NameVal = HKObj->getString("name");
    if (!NameVal)
      continue;

    KernelConstProfile KP;
    KP.name = *NameVal;

    // Parse common_scalars
    json::Array *ScalarsArr = HKObj->getArray("common_scalars");
    if (ScalarsArr) {
      for (json::Value &ScalarVal : *ScalarsArr) {
        json::Object *ScalarObj = ScalarVal.getAsObject();
        if (!ScalarObj)
          continue;
        
        ScalarConstInfo SCI;
        if (auto ArgVal = ScalarObj->get("arg")) {
          if (auto ArgInt = ArgVal->getAsInteger()) {
            SCI.index = static_cast<unsigned>(*ArgInt);
          } else {
            continue;
          }
        } else {
          continue;
        }
        
        if (auto ValueVal = ScalarObj->getString("value")) {
          SCI.value = *ValueVal;
        } else {
          continue;
        }
        
        if (auto RatioVal = ScalarObj->get("ratio")) {
          if (auto RatioNum = RatioVal->getAsNumber()) {
            SCI.ratio = *RatioNum;
          } else {
            SCI.ratio = 1.0;
          }
        } else {
          SCI.ratio = 1.0;
        }
        
        KP.commonScalars.push_back(SCI);
      }
    }

    // Parse common_dims
    json::Array *DimsArr = HKObj->getArray("common_dims");
    if (DimsArr) {
      for (json::Value &DimVal : *DimsArr) {
        json::Object *DimObj = DimVal.getAsObject();
        if (!DimObj)
          continue;
        
        DimConstInfo DCI;
        if (auto DimVal = DimObj->getString("dim")) {
          DCI.dim = *DimVal;
        } else {
          continue;
        }
        
        if (auto ValueVal = DimObj->get("value")) {
          if (auto ValueInt = ValueVal->getAsInteger()) {
            DCI.value = *ValueInt;
          } else {
            continue;
          }
        } else {
          continue;
        }
        
        if (auto RatioVal = DimObj->get("ratio")) {
          if (auto RatioNum = RatioVal->getAsNumber()) {
            DCI.ratio = *RatioNum;
          } else {
            DCI.ratio = 1.0;
          }
        } else {
          DCI.ratio = 1.0;
        }
        
        KP.commonDims.push_back(DCI);
      }
    }

    if (!KP.commonScalars.empty() || !KP.commonDims.empty()) {
      KernelProfiles[KP.name] = std::move(KP);
    }
  }

  return true;
}

// Check if a parameter is used in control flow (loops, branches, etc.)
bool CudaKernelConstPass::isUsedInControlFlow(const Function &F, unsigned paramIndex) {
  if (paramIndex >= F.arg_size())
    return false;
  
  const Argument *Arg = F.getArg(paramIndex);
  return isUsedInLoop(F, paramIndex) || isUsedInBranch(F, paramIndex);
}

// Check if a parameter is used in loop conditions or bounds
bool CudaKernelConstPass::isUsedInLoop(const Function &F, unsigned paramIndex) {
  if (paramIndex >= F.arg_size())
    return false;
  
  const Argument *Arg = F.getArg(paramIndex);
  
  // Check all uses of this argument
  for (const User *U : Arg->users()) {
    if (const Instruction *I = dyn_cast<Instruction>(U)) {
      // Check if used in comparison instructions (common in loop conditions)
      if (isa<ICmpInst>(I) || isa<FCmpInst>(I)) {
        // Check if this comparison is used in a branch instruction
        for (const User *CmpUser : I->users()) {
          if (isa<BranchInst>(CmpUser)) {
            return true;
          }
        }
      }
      
      // Check if used in PHI nodes (common in loop induction variables)
      if (isa<PHINode>(I)) {
        return true;
      }
    }
  }
  
  return false;
}

// Check if a parameter is used in branch conditions
bool CudaKernelConstPass::isUsedInBranch(const Function &F, unsigned paramIndex) {
  if (paramIndex >= F.arg_size())
    return false;
  
  const Argument *Arg = F.getArg(paramIndex);
  
  // Check all uses of this argument
  for (const User *U : Arg->users()) {
    if (const Instruction *I = dyn_cast<Instruction>(U)) {
      // Check if used directly in branch instructions
      if (const BranchInst *BI = dyn_cast<BranchInst>(I)) {
        if (BI->isConditional()) {
          return true;
        }
      }
      
      // Check if used in comparison instructions
      if (isa<ICmpInst>(I) || isa<FCmpInst>(I)) {
        // Check if this comparison is used in a branch instruction
        for (const User *CmpUser : I->users()) {
          if (const BranchInst *BI = dyn_cast<BranchInst>(CmpUser)) {
            if (BI->isConditional()) {
              return true;
            }
          }
        }
      }
    }
  }
  
  return false;
}

// Check if this kernel should be optimized
bool CudaKernelConstPass::shouldOptimizeKernel(const Function &F) {
  auto It = KernelProfiles.find(F.getName().str());
  if (It == KernelProfiles.end())
    return false;
  
  const KernelConstProfile &Profile = It->second;
  
  // Check if any common scalar is used in control flow
  for (const ScalarConstInfo &SCI : Profile.commonScalars) {
    if (isUsedInControlFlow(F, SCI.index)) {
      return true;
    }
  }
  
  // TODO: Add dimension checking logic if needed
  
  return false;
}

// Create a constant value from string representation
Value *CudaKernelConstPass::getConstantValue(const std::string &valueStr, Type *type) {
  if (type->isIntegerTy()) {
    int64_t val = std::stoll(valueStr);
    return ConstantInt::get(type, val);
  } else if (type->isFloatingPointTy()) {
    double val = std::stod(valueStr);
    return ConstantFP::get(type, val);
  }
  
  return nullptr;
}

// Replace all uses of a parameter with a constant value
void CudaKernelConstPass::replaceUsesWithConstant(Function &F, unsigned paramIndex, 
                                                  Value *constantValue) {
  if (paramIndex >= F.arg_size())
    return;
  
  Argument *Arg = F.getArg(paramIndex);
  Arg->replaceAllUsesWith(constantValue);
}

// Perform constant propagation on the cloned function
void CudaKernelConstPass::performConstantPropagation(Function &F, 
                                                     const KernelConstProfile &Profile) {
  for (const ScalarConstInfo &SCI : Profile.commonScalars) {
    if (SCI.index < F.arg_size()) {
      Argument *Arg = F.getArg(SCI.index);
      Value *ConstVal = getConstantValue(SCI.value, Arg->getType());
      if (ConstVal) {
        replaceUsesWithConstant(F, SCI.index, ConstVal);
      }
    }
  }
  
  // TODO: Add dimension constant propagation logic if needed
}

// Clone kernel with constant propagation
Function *CudaKernelConstPass::cloneKernelWithConstProp(Function &OrigF, 
                                                       const KernelConstProfile &Profile) {
  ValueToValueMapTy VMap;
  Function *ClonedF = CloneFunction(&OrigF, VMap);
  
  // Set new function name for the cloned variant
  std::string NewName = OrigF.getName().str() + "_const";
  ClonedF->setName(NewName);
  
  // Perform constant propagation on the cloned kernel
  performConstantPropagation(*ClonedF, Profile);
  
  // Copy the existing nvvm.annotations entry so the clone is still a kernel
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

PreservedAnalyses CudaKernelConstPass::run(Module &M, ModuleAnalysisManager &AM) {
  if (!parseProfileLog()) {
    errs() << "Warning: Failed to parse profile log, skipping constant propagation optimization\n";
    return PreservedAnalyses::all();
  }

  bool Changed = false;
  std::vector<Function *> KernelsToProcess;
  
  // Collect all kernel functions in the module
  for (Function &F : M) {
    if (isKernelFunction(F)) {
      KernelsToProcess.push_back(&F);
    }
  }
  
  // Process each kernel function
  for (Function *F : KernelsToProcess) {
    if (shouldOptimizeKernel(*F)) {
      auto It = KernelProfiles.find(F->getName().str());
      if (It != KernelProfiles.end()) {
        Function *ClonedF = cloneKernelWithConstProp(*F, It->second);
        Changed = true;
        
        LLVM_DEBUG(dbgs() << "Cloned kernel " << F->getName() 
                         << " to " << ClonedF->getName() 
                         << " with constant propagation for " 
                         << It->second.commonScalars.size() 
                         << " scalar parameters\n");
      }
    }
  }
  
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
} 