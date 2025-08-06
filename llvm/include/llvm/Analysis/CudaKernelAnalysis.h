
#ifndef LLVM_ANALYSIS_CUDAKERNELANALYSIS_H
#define LLVM_ANALYSIS_CUDAKERNELANALYSIS_H

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {

// The result of the analysis
class CudaKernelAnalysisResult {
public:
  CudaKernelAnalysisResult(DenseMap<const Argument *, int> ScalarWeights,
                           DenseMap<const Argument *, bool> PointerBenefits,
                           bool DebugOutput = false)
      : ScalarWeights(std::move(ScalarWeights)),
        PointerBenefits(std::move(PointerBenefits)),
        DebugOutput(DebugOutput) {}

  // A map from scalar argument to its weight based on control-flow usage.
  DenseMap<const Argument *, int> ScalarWeights;

  // A map from pointer argument to a boolean indicating if adding 'noalias'
  // is likely beneficial.
  DenseMap<const Argument *, bool> PointerBenefits;

  // Debug output flag
  bool DebugOutput;

  void print(raw_ostream &OS) const;
};

// The analysis pass itself
class CudaKernelAnalysis : public AnalysisInfoMixin<CudaKernelAnalysis> {
public:
  using Result = CudaKernelAnalysisResult;

  Result run(Function &F, FunctionAnalysisManager &AM);

private:
  friend AnalysisInfoMixin<CudaKernelAnalysis>;
  static AnalysisKey Key;
};

} // namespace llvm

#endif // LLVM_ANALYSIS_CUDAKERNELANALYSIS_H
