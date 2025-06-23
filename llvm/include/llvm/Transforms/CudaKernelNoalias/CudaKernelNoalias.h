//===-- CudaKernelNoalias.h - CUDA Kernel Noalias Optimization ----------===//
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

#ifndef LLVM_TRANSFORMS_CUDAKERNELNOALIAS_CUDAKERNELNOALIAS_H
#define LLVM_TRANSFORMS_CUDAKERNELNOALIAS_CUDAKERNELNOALIAS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Module;

/// Pass that clones CUDA kernel functions to create noalias versions
class CudaKernelNoaliasPass : public PassInfoMixin<CudaKernelNoaliasPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_CUDAKERNELNOALIAS_CUDAKERNELNOALIAS_H 