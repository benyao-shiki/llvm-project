//===- CudaArgsProfile.cpp - CUDA Kernel Arguments Profiling Pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass collects type information for CUDA kernel arguments and embeds it
// as a JSON string in the final binary using the llvm.nvvm.reflect intrinsic.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/CudaArgsProfile/CudaArgsProfile.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <string>

using namespace llvm;

namespace {

class CudaArgsProfileImpl {
private:
  Module *M;
  LLVMContext *Context;

  std::string getTypeString(Type *T) {
    std::string TypeName;
    raw_string_ostream RSO(TypeName);
    T->print(RSO);
    return RSO.str();
  }

  std::string getStructMemberTypes(StructType *ST) {
    if (ST->isOpaque()) {
      return "opaque";
    }
    std::string MemberTypes;
    for (unsigned i = 0, e = ST->getNumElements(); i < e; ++i) {
      if (i > 0) {
        MemberTypes += ", ";
      }
      MemberTypes += getTypeString(ST->getElementType(i));
    }
    return MemberTypes;
  }

public:
  CudaArgsProfileImpl(Module &Mod) : M(&Mod), Context(&Mod.getContext()) {}

  bool runOnModule() {
    std::string JsonString = "{\"kernels\":[";
    bool FirstKernel = true;

    for (Function &F : *M) {
      if (F.getCallingConv() != CallingConv::PTX_Kernel) {
        continue;
      }

      if (!FirstKernel) {
        JsonString += ",";
      }
      FirstKernel = false;

      JsonString += "{\"name\":\"" + F.getName().str() + "\",";

      std::string LineInfo;
      for (Argument &Arg : F.args()) {
        LineInfo += getTypeString(Arg.getType());

        if (Arg.hasByValAttr()) {
          Type *ByValTy = Arg.getParamByValType();
          if (StructType *ST = dyn_cast<StructType>(ByValTy)) {
            LineInfo += " (struct";
            if (ST->hasName()) {
              LineInfo += " " + ST->getName().str();
            }
            LineInfo += ": {" + getStructMemberTypes(ST) + "})";
          }
        }
        LineInfo += "; ";
      }

      if (!LineInfo.empty()) {
        LineInfo.pop_back(); // remove trailing space
        LineInfo.pop_back(); // remove trailing ;
      }

      JsonString += "\"param_types\":\"" + LineInfo + "\"}";
    }

    JsonString += "]}";

    if (FirstKernel) { // No kernels found
      return false;
    }

    auto *JsonConst = ConstantDataArray::getString(*Context, JsonString, true);
    auto *GV = new GlobalVariable(*M, JsonConst->getType(), true,
                                GlobalValue::PrivateLinkage, JsonConst,
                                "__nv_kernel_param_types_data__");
    GV->setAlignment(Align(1));

    Function *CtorF = Function::Create(
        FunctionType::get(Type::getVoidTy(*Context), false),
        GlobalValue::InternalLinkage, "__cuda_profile_module_ctor", M);
    BasicBlock *CtorBB = BasicBlock::Create(*Context, "entry", CtorF);
    IRBuilder<> Builder(CtorBB);

    FunctionCallee ReflectFunc =
        Intrinsic::getOrInsertDeclaration(M, Intrinsic::nvvm_reflect);

    Value *GVPtr = Builder.CreateConstInBoundsGEP2_32(GV->getValueType(), GV, 0, 0);
    Builder.CreateCall(ReflectFunc, {GVPtr});
    Builder.CreateRetVoid();

    appendToGlobalCtors(*M, CtorF, 0);

    return true;
  }
};

} // anonymous namespace

PreservedAnalyses CudaArgsProfilePass::run(Module &M,
                                           ModuleAnalysisManager &AM) {
  CudaArgsProfileImpl Impl(M);
  if (Impl.runOnModule()) {
    return PreservedAnalyses::none();
  }
  return PreservedAnalyses::all();
}
