
#ifndef LLVM_ANALYSIS_CUDAKERNELANALYSIS_H
#define LLVM_ANALYSIS_CUDAKERNELANALYSIS_H

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace llvm {

// Structure to represent parameter information (both regular arguments and struct members)
struct ParameterInfo {
  std::string Name;                    // Parameter name
  const Argument *Arg;                 // Pointer to the original argument
  SmallVector<unsigned, 4> Indices;    // Member indices (length 1 for regular args, >1 for struct members)
  
  ParameterInfo(const std::string &N, const Argument *A, unsigned ArgIndex)
      : Name(N), Arg(A) {
    Indices.push_back(ArgIndex);
  }
  
  ParameterInfo(const std::string &N, const Argument *A, 
                const SmallVector<unsigned, 4> &Idx)
      : Name(N), Arg(A), Indices(Idx) {}
  
  // For use as key in maps
  bool operator<(const ParameterInfo &Other) const {
    if (Name != Other.Name) return Name < Other.Name;
    if (Arg != Other.Arg) return Arg < Other.Arg;
    if (Indices.size() != Other.Indices.size()) return Indices.size() < Other.Indices.size();
    for (size_t i = 0; i < Indices.size(); ++i) {
      if (Indices[i] != Other.Indices[i]) return Indices[i] < Other.Indices[i];
    }
    return false;
  }
  
  bool operator==(const ParameterInfo &Other) const {
    return Name == Other.Name && Arg == Other.Arg && Indices == Other.Indices;
  }
  
  // Get a string representation for debugging
  std::string getDisplayName() const {
    if (Indices.size() <= 1) {
      return Name;
    }
    std::string Result = Name;
    for (size_t i = 1; i < Indices.size(); ++i) {
      Result += "[" + std::to_string(Indices[i]) + "]";
    }
    return Result;
  }
  
  // Check if this is a struct member (indices length > 1)
  bool isStructMember() const {
    return Indices.size() > 1;
  }
  
  // Get the argument index (first element of indices)
  unsigned getArgIndex() const {
    return Indices.empty() ? 0 : Indices[0];
  }
};

// The result of the analysis
class CudaKernelAnalysisResult {
public:
  CudaKernelAnalysisResult(DenseMap<ParameterInfo, int> ScalarWeights,
                           DenseMap<ParameterInfo, int> PointerWeights,
                           bool DebugOutput = false)
      : ScalarWeights(std::move(ScalarWeights)),
        PointerWeights(std::move(PointerWeights)),
        DebugOutput(DebugOutput) {}

  // A map from scalar parameter info to its weight
  DenseMap<ParameterInfo, int> ScalarWeights;

  // A map from pointer parameter info to its weight
  DenseMap<ParameterInfo, int> PointerWeights;

  // Debug output flag
  bool DebugOutput;

  void print(raw_ostream &OS) const;
  
  // Helper method to get all parameters sorted by weight
  SmallVector<std::pair<ParameterInfo, int>, 16> getAllParametersSorted() const;
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

// DenseMapInfo specialization for ParameterInfo
namespace llvm {
template<> struct DenseMapInfo<ParameterInfo> {
  static inline ParameterInfo getEmptyKey() {
    return ParameterInfo("", nullptr, 0);
  }
  
  static inline ParameterInfo getTombstoneKey() {
    return ParameterInfo("", nullptr, 1);
  }
  
  static unsigned getHashValue(const ParameterInfo &Val) {
    uint64_t Hash = hash_value(Val.Name);
    Hash = hash_combine(Hash, Val.Arg);
    // Hash the indices manually since SmallVector doesn't have hash_value
    for (unsigned Idx : Val.Indices) {
      Hash = hash_combine(Hash, Idx);
    }
    return (unsigned)Hash;
  }
  
  static bool isEqual(const ParameterInfo &LHS, const ParameterInfo &RHS) {
    return LHS == RHS;
  }
};
}

#endif // LLVM_ANALYSIS_CUDAKERNELANALYSIS_H
