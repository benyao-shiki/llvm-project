#include "llvm/Analysis/CudaKernelAnalysis.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "cuda-kernel-analysis"

// Heuristic to determine if a pointer argument benefits from 'noalias'
// A simple heuristic: if the pointer is used inside a loop, it's likely beneficial.
bool isNoAliasBeneficial(const Argument &A, Function &F, FunctionAnalysisManager &AM) {
  if (!A.getType()->isPointerTy()) return false;

  auto &LI = AM.getResult<LoopAnalysis>(F);
  for (const User *U : A.users()) {
    if (const Instruction *I = dyn_cast<Instruction>(U)) {
      if (LI.getLoopFor(I->getParent())) {
        return true; // Used inside a loop
      }
    }
  }
  return false;
}

CudaKernelAnalysis::Result CudaKernelAnalysis::run(Function &F, FunctionAnalysisManager &AM) {
  DenseMap<const Argument *, int> ScalarWeights;
  DenseMap<const Argument *, bool> PointerBenefits;

  // This analysis is only for CUDA kernels.
  // We can identify them by calling convention or by the "ptx.kernel" attribute.
  if (F.getCallingConv() != CallingConv::PTX_Kernel && !F.hasFnAttribute("ptx.kernel")) {
      LLVM_DEBUG(dbgs() << "CudaKernelAnalysis: Function '" << F.getName() 
                       << "' is not a CUDA kernel, skipping analysis\n");
      return CudaKernelAnalysisResult(ScalarWeights, PointerBenefits);
  }

  LLVM_DEBUG(dbgs() << "CudaKernelAnalysis: Analyzing CUDA kernel function '" 
                   << F.getName() << "'\n");

  for (const Argument &A : F.args()) {
    if (A.getType()->isPointerTy()) {
      bool benefit = isNoAliasBeneficial(A, F, AM);
      PointerBenefits[&A] = benefit;
      LLVM_DEBUG(dbgs() << "  Pointer argument '" << A.getName() 
                       << "': noalias benefit = " << (benefit ? "true" : "false") << "\n");
    } else if (A.getType()->isIntegerTy() || A.getType()->isFloatingPointTy()) {
      int weight = 0; 
      for (const User *U : A.users()) {
        if (const Instruction *I = dyn_cast<Instruction>(U)) {
          if (isa<BranchInst>(I) || isa<SwitchInst>(I)) {
            weight += 10; // Higher weight for direct use in branches
          } else if (isa<CmpInst>(I)) {
            // If the result of a comparison is used in a branch, it's important.
            for (const User *CmpUser : I->users()) {
                if (const Instruction *CmpI = dyn_cast<Instruction>(CmpUser)) {
                    if(isa<BranchInst>(CmpI) || isa<SwitchInst>(CmpI)) {
                        weight += 5;
                    }
                }
            }
          }
        }
      }
      ScalarWeights[&A] = weight;
      LLVM_DEBUG(dbgs() << "  Scalar argument '" << A.getName() 
                       << "': weight = " << weight << "\n");
    }
  }

  CudaKernelAnalysisResult Result(std::move(ScalarWeights), std::move(PointerBenefits));
  
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
  OS << "  Scalar Argument Weights:\n";
  for (auto const& [Arg, Weight] : ScalarWeights) {
    OS << "    - " << Arg->getName() << ": " << Weight << "\n";
  }
  OS << "  Pointer Argument NoAlias Benefits:\n";
  for (auto const& [Arg, Benefit] : PointerBenefits) {
    OS << "    - " << Arg->getName() << ": " << (Benefit ? "true" : "false") << "\n";
  }
}

AnalysisKey CudaKernelAnalysis::Key;

