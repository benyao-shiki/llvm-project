//===- CudaArgsProfile.h - CUDA Kernel Arguments Profiling Pass ---------===//
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

#ifndef LLVM_TRANSFORMS_CUDAARGSPROFILE_CUDAARGSPROFILE_H
#define LLVM_TRANSFORMS_CUDAARGSPROFILE_CUDAARGSPROFILE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Module;

class CudaArgsProfilePass : public PassInfoMixin<CudaArgsProfilePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_CUDAARGSPROFILE_CUDAARGSPROFILE_H 