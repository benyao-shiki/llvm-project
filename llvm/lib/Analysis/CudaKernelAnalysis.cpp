#include "llvm/Analysis/CudaKernelAnalysis.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/STLExtras.h"

using namespace llvm;

#define DEBUG_TYPE "cuda-kernel-analysis"

// Generic function to analyze scalar values (arguments or extracted values)
int analyzeScalarValue(const Value *V, Function &F, FunctionAnalysisManager &AM) {
  if (!V->getType()->isIntegerTy() && !V->getType()->isFloatingPointTy()) {
    return 0;
  }

  int weight = 0;
  for (const User *U : V->users()) {
    if (const Instruction *I = dyn_cast<Instruction>(U)) {
      if (isa<BranchInst>(I) || isa<SwitchInst>(I)) {
        weight += 10; // Higher weight for direct use in branches
      } else if (isa<CmpInst>(I)) {
        // If the result of a comparison is used in a branch, it's important.
        for (const User *CmpUser : I->users()) {
          if (const Instruction *CmpI = dyn_cast<Instruction>(CmpUser)) {
            if (isa<BranchInst>(CmpI) || isa<SwitchInst>(CmpI)) {
              weight += 5;
            }
          }
        }
      }
    }
  }
  return weight;
}

// Generic function to analyze pointer values (arguments or extracted values)
int analyzePointerValue(const Value *V, Function &F, FunctionAnalysisManager &AM) {
  if (!V->getType()->isPointerTy()) {
    return 0;
  }

  int weight = 0;
  auto &LI = AM.getResult<LoopAnalysis>(F);
  
  for (const User *U : V->users()) {
    if (const Instruction *I = dyn_cast<Instruction>(U)) {
      // Base weight for being used in the function
      weight += 1;
      
      // Higher weight if used inside a loop
      if (LI.getLoopFor(I->getParent())) {
        weight += 5;
      }
      
      // Additional weight for specific instruction types that benefit from noalias
      if (isa<LoadInst>(I) || isa<StoreInst>(I)) {
        weight += 3; // Memory operations benefit more from noalias
      } else if (isa<GetElementPtrInst>(I)) {
        weight += 2; // Pointer arithmetic operations
      }
      
      // Check if the pointer is used in loop-invariant ways
      if (const Loop *L = LI.getLoopFor(I->getParent())) {
        if (L->isLoopInvariant(V)) {
          weight += 2; // Loop-invariant pointers are good candidates for noalias
        }
      }
    }
  }
  
  return weight;
}

// Analyze struct members recursively and add them to the weights maps
void analyzeStructMembers(const Argument &A, unsigned ArgIndex, Function &F, FunctionAnalysisManager &AM,
                         DenseMap<ParameterInfo, int> &ScalarWeights,
                         DenseMap<ParameterInfo, int> &PointerWeights,
                         const SmallVector<unsigned, 4> &ParentIndices = {}) {
  if (!A.getType()->isStructTy()) {
    return;
  }

  StructType *ST = cast<StructType>(A.getType());
  
  // Analyze each struct member
  for (unsigned i = 0; i < ST->getNumElements(); ++i) {
    Type *MemberType = ST->getElementType(i);
    
    // Create indices for this member
    SmallVector<unsigned, 4> MemberIndices = ParentIndices;
    if (MemberIndices.empty()) {
      MemberIndices.push_back(ArgIndex); // Add argument index as first element
    }
    MemberIndices.push_back(i);
    
    // Look for extractvalue instructions that access this member
    for (const User *U : A.users()) {
      if (const ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(U)) {
        if (EVI->getIndices().size() == MemberIndices.size() - 1 &&
            std::equal(EVI->getIndices().begin(), EVI->getIndices().end(), 
                      MemberIndices.begin() + 1)) {
          // This extractvalue accesses our member
          
          // Analyze scalar members
          if (MemberType->isIntegerTy() || MemberType->isFloatingPointTy()) {
            int weight = analyzeScalarValue(EVI, F, AM);
            if (weight > 0) {
              ParameterInfo ParamInfo(A.getName().str(), &A, MemberIndices);
              ScalarWeights[ParamInfo] = weight;
            }
          }
          
          // Analyze pointer members
          if (MemberType->isPointerTy()) {
            int weight = analyzePointerValue(EVI, F, AM);
            if (weight > 0) {
              ParameterInfo ParamInfo(A.getName().str(), &A, MemberIndices);
              PointerWeights[ParamInfo] = weight;
            }
          }
          
          // Recursively analyze nested struct members
          if (MemberType->isStructTy()) {
            analyzeStructMembers(A, ArgIndex, F, AM, ScalarWeights, PointerWeights, MemberIndices);
          }
          
          break; // Found the extractvalue for this member, no need to continue
        }
      }
    }
  }
}

CudaKernelAnalysis::Result CudaKernelAnalysis::run(Function &F, FunctionAnalysisManager &AM) {
  DenseMap<ParameterInfo, int> ScalarWeights;
  DenseMap<ParameterInfo, int> PointerWeights;

  // This analysis is only for CUDA kernels.
  // We can identify them by calling convention or by the "ptx.kernel" attribute.
  if (F.getCallingConv() != CallingConv::PTX_Kernel && !F.hasFnAttribute("ptx.kernel")) {
      LLVM_DEBUG(dbgs() << "CudaKernelAnalysis: Function '" << F.getName() 
                       << "' is not a CUDA kernel, skipping analysis\n");
      return CudaKernelAnalysisResult(ScalarWeights, PointerWeights);
  }

  LLVM_DEBUG(dbgs() << "CudaKernelAnalysis: Analyzing CUDA kernel function '" 
                   << F.getName() << "'\n");

  unsigned ArgIndex = 0;
  for (const Argument &A : F.args()) {
    if (A.getType()->isPointerTy()) {
      int weight = analyzePointerValue(&A, F, AM);
      ParameterInfo ParamInfo(A.getName().str(), &A, ArgIndex);
      PointerWeights[ParamInfo] = weight;
      LLVM_DEBUG(dbgs() << "  Pointer argument '" << A.getName() 
                       << "': weight = " << weight << "\n");
    } else if (A.getType()->isIntegerTy() || A.getType()->isFloatingPointTy()) {
      int weight = analyzeScalarValue(&A, F, AM);
      ParameterInfo ParamInfo(A.getName().str(), &A, ArgIndex);
      ScalarWeights[ParamInfo] = weight;
      LLVM_DEBUG(dbgs() << "  Scalar argument '" << A.getName() 
                       << "': weight = " << weight << "\n");
    } else if (A.getType()->isStructTy()) {
      // Analyze struct members
      analyzeStructMembers(A, ArgIndex, F, AM, ScalarWeights, PointerWeights);
      LLVM_DEBUG(dbgs() << "  Struct argument '" << A.getName() 
                       << "': analyzing " << cast<StructType>(A.getType())->getNumElements() 
                       << " members\n");
    }
    ArgIndex++;
  }

  CudaKernelAnalysisResult Result(std::move(ScalarWeights), std::move(PointerWeights));
  
  // Debug output if requested
  if (Result.DebugOutput) {
    errs() << "CudaKernelAnalysis for function '" << F.getName() << "':\n";
    Result.print(errs());
  }

  LLVM_DEBUG(dbgs() << "CudaKernelAnalysis: Analysis completed for function '" 
                   << F.getName() << "'\n");

  return Result;
}

void CudaKernelAnalysisResult::print(raw_ostream &OS) const {
  OS << "  Scalar Parameter Weights:\n";
  for (auto const& [Param, Weight] : ScalarWeights) {
    OS << "    - " << Param.getDisplayName() << ": " << Weight << "\n";
  }
  OS << "  Pointer Parameter Weights:\n";
  for (auto const& [Param, Weight] : PointerWeights) {
    OS << "    - " << Param.getDisplayName() << ": " << Weight << "\n";
  }
}

SmallVector<std::pair<ParameterInfo, int>, 16> 
CudaKernelAnalysisResult::getAllParametersSorted() const {
  SmallVector<std::pair<ParameterInfo, int>, 16> Result;
  
  // Add scalar parameters
  for (auto const& [Param, Weight] : ScalarWeights) {
    Result.emplace_back(Param, Weight);
  }
  
  // Add pointer parameters
  for (auto const& [Param, Weight] : PointerWeights) {
    Result.emplace_back(Param, Weight);
  }
  
  // Sort by weight in descending order
  llvm::sort(Result, [](const auto &A, const auto &B) {
    return A.second > B.second;
  });
  
  return Result;
}

AnalysisKey CudaKernelAnalysis::Key;

