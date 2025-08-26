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
#include "llvm/ADT/StringMap.h"
#include <string>
#include <map>
#include <vector>

namespace llvm {

class Module;

// Per-kernel profile info for noalias constructed from launches[]
struct KernelNoaliasProfile {
  std::string name;
  std::map<unsigned, std::string> argIndexToName; // readable names (optional)
  std::vector<StringMap<int>> launches;           // flattened path -> alias flag (0 means no-alias)
  // Map from host stub names to device side names for accurate kernel matching
  std::map<std::string, std::string> deviceSideNames;
};

// Selected item to mark as noalias (full indices path)
struct SelectedNoaliasItem {
  SmallVector<unsigned, 4> Indices; // [arg, member, ...]
  double Ratio = 0.0;               // single ratio for reporting
};

/// Pass that clones CUDA kernel functions to create noalias versions
class CudaKernelNoaliasPass : public PassInfoMixin<CudaKernelNoaliasPass> {
public:
  CudaKernelNoaliasPass(std::string profilePath = "") : ProfilePath(profilePath) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  std::string ProfilePath;
  std::map<std::string, KernelNoaliasProfile> KernelProfiles;
  std::map<std::string, std::vector<SelectedNoaliasItem>> SelectedByKernel;

  bool parseProfileLog();
  void computeBestSelections(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_CUDAKERNELNOALIAS_CUDAKERNELNOALIAS_H 