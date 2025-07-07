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
#include <string>
#include <map>
#include <vector>

namespace llvm {

// 存储指针参数的信息
struct PointerInfo {
  unsigned index; // 参数索引 (0-based)
};

// 存储kernel的profile信息
struct KernelProfile {
  std::string name;
  std::vector<PointerInfo> pointerParams;
};

class Module;

/// Pass that clones CUDA kernel functions to create noalias versions
class CudaKernelNoaliasPass : public PassInfoMixin<CudaKernelNoaliasPass> {
public:
  CudaKernelNoaliasPass(std::string profilePath = "") : ProfilePath(profilePath) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  std::string ProfilePath;
  std::map<std::string, KernelProfile> KernelProfiles;
  
  bool parseProfileLog();
  std::vector<unsigned> getNoAliasPointerIndices(const Function &F);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_CUDAKERNELNOALIAS_CUDAKERNELNOALIAS_H 