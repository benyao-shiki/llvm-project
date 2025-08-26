//===-- CudaKernelConst.h - CUDA Kernel Const Propagation ---------------===//
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

#ifndef LLVM_TRANSFORMS_CUDAKERNELCONST_CUDAKERNELCONST_H
#define LLVM_TRANSFORMS_CUDAKERNELCONST_CUDAKERNELCONST_H

#include "llvm/IR/PassManager.h"
#include <string>
#include <map>
#include <vector>
#include "llvm/ADT/ArrayRef.h"

namespace llvm {

// Information for a scalar argument that has a common value
struct ScalarConstInfo {
  unsigned index;     // Argument index (0-based)
  std::string value;  // Common constant value as string
  double ratio;       // Frequency ratio (0.0 to 1.0)
};

// Shared struct used to serialize selected parameters with full indices
struct SelectedParamRecord {
  std::vector<unsigned> Indices;
  std::string Value;
  double Ratio;
};

// Information for a dimension that has a common value
struct DimConstInfo {
  std::string dim;    // Dimension name (e.g., "gridDim.x", "blockDim.x")
  int64_t value;      // Common constant value
  double ratio;       // Frequency ratio (0.0 to 1.0)
};

// Per-kernel profile information for constant propagation
struct KernelConstProfile {
  std::string name;
  std::vector<ScalarConstInfo> commonScalars;
  std::vector<DimConstInfo> commonDims;
  // All recorded launches: map from index-path string (e.g. "2" or "1.0.2") to its value as string
  std::vector<std::map<std::string, std::string>> launches;
  // Optional: map from flat argument index to the stable name from profile
  std::map<unsigned, std::string> argIndexToName;
  // Map from host stub names to device side names for accurate kernel matching
  std::map<std::string, std::string> deviceSideNames;
};

class Module;
class Function;
class BasicBlock;
class Value;
class Instruction;
class Type;

/// Pass that clones CUDA kernel functions to create constant propagation versions
class CudaKernelConstPass : public PassInfoMixin<CudaKernelConstPass> {
public:
  CudaKernelConstPass(std::string profilePath = "") : ProfilePath(profilePath) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  std::string ProfilePath;
  std::map<std::string, KernelConstProfile> KernelProfiles;
  // Best selected scalar constants per kernel name, computed using analysis weights and profile frequencies
  std::map<std::string, std::vector<ScalarConstInfo>> SelectedScalarsByKernel;
  std::map<std::string, std::vector<SelectedParamRecord>> SelectedParamsByKernel;
  // Compute best parameter combinations using CudaKernelAnalysis and profile ratios
  void computeBestSelections(Module &M, ModuleAnalysisManager &AM);
  
  bool parseProfileLog();
  bool shouldOptimizeKernel(const Function &F);
  bool isUsedInControlFlow(const Function &F, unsigned paramIndex);
  bool isUsedInLoop(const Function &F, unsigned paramIndex);
  bool isUsedInBranch(const Function &F, unsigned paramIndex);
  Function *cloneKernelWithConstProp(Function &OrigF,
                                    const KernelConstProfile &Profile,
                                    ArrayRef<SelectedParamRecord> SPs);
  void performConstantPropagation(Function &F,
                                    const KernelConstProfile &Profile,
                                    ArrayRef<SelectedParamRecord> SPs);
  Value *getConstantValue(const std::string &valueStr, Type *type);
  void replaceUsesWithConstant(Function &F, unsigned paramIndex, 
                               Value *constantValue);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_CUDAKERNELCONST_CUDAKERNELCONST_H 