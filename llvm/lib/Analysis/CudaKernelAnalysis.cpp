#include "llvm/Analysis/CudaKernelAnalysis.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Attributes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/InstIterator.h"

using namespace llvm;

// Special indices for implicit CUDA parameters
// These are chosen to be large values to avoid collision with real argument indices.
// The values must be kept in sync with CudaKernelConst.cpp
namespace {
static constexpr unsigned SPECIAL_INDEX_BASE = 0xE0000000;
static constexpr unsigned GRID_DIM_X_INDEX  = SPECIAL_INDEX_BASE;
static constexpr unsigned GRID_DIM_Y_INDEX  = SPECIAL_INDEX_BASE - 1;
static constexpr unsigned GRID_DIM_Z_INDEX  = SPECIAL_INDEX_BASE - 2;
static constexpr unsigned BLOCK_DIM_X_INDEX = SPECIAL_INDEX_BASE - 3;
static constexpr unsigned BLOCK_DIM_Y_INDEX = SPECIAL_INDEX_BASE - 4;
static constexpr unsigned BLOCK_DIM_Z_INDEX = SPECIAL_INDEX_BASE - 5;
}

#define DEBUG_TYPE "cuda-kernel-analysis"

// fwd-declarations
static int analyzeScalarValue(const Value *V, Function &F, FunctionAnalysisManager &AM);
static int analyzePointerValue(const Value *V, Function &F, FunctionAnalysisManager &AM);
static void analyzeStructMembers(const Argument &A, unsigned ArgIndex, Function &F, FunctionAnalysisManager &AM,
                         DenseMap<ParameterInfo, int> &ScalarWeights,
                         DenseMap<ParameterInfo, int> &PointerWeights,
                         const SmallVector<unsigned, 4> &ParentIndices = {});
static void traversePointerMembers(const Argument &A,
                                   Value *PtrV,
                                   uint64_t CurrentOffset,
                                   Function &F,
                                   FunctionAnalysisManager &AM,
                                   DenseMap<ParameterInfo, int> &ScalarWeights,
                                   DenseMap<ParameterInfo, int> &PointerWeights);


// Recursively traverse GEPs and loads to identify and weigh struct members.
static void traversePointerMembers(const Argument &A,
                                   Value *PtrV,
                                   uint64_t CurrentOffset,
                                   Function &F,
                                   FunctionAnalysisManager &AM,
                                   DenseMap<ParameterInfo, int> &ScalarWeights,
                                   DenseMap<ParameterInfo, int> &PointerWeights) {
  const DataLayout &DL = F.getParent()->getDataLayout();

  for (User *U : PtrV->users()) {
    if (auto *LI = dyn_cast<LoadInst>(U)) {
      if (LI->getType()->isPointerTy()) {
        int W = analyzePointerValue(LI, F, AM);
        if (W > 0) {
          SmallVector<unsigned, 4> MemberPath;
          MemberPath.push_back(A.getArgNo());
          MemberPath.push_back(CurrentOffset); // Should be 0 for the first member
          ParameterInfo P(A.getName().str(), &A, MemberPath);
          auto It = PointerWeights.find(P);
          if (It == PointerWeights.end() || It->second < W)
            PointerWeights[P] = W;
        }
        traversePointerMembers(A, LI, CurrentOffset, F, AM, ScalarWeights, PointerWeights);
      }
      continue;
    }

    auto *GEP = dyn_cast<GetElementPtrInst>(U);
    if (!GEP) continue;

    // We need the offset relative to the GEP's base pointer (PtrV).
    APInt RelativeOffset(DL.getPointerSizeInBits(GEP->getPointerAddressSpace()), 0);
    if (!GEP->accumulateConstantOffset(DL, RelativeOffset)) {
      //print the non-constant GEP for debug
      LLVM_DEBUG(dbgs() << "  GEP with non-constant offset: " << *GEP << "\n");
      continue; // Skip GEPs with non-constant offsets.
    }

    uint64_t TotalOffset = CurrentOffset + RelativeOffset.getZExtValue();

    Type *Ty = GEP->getResultElementType();
    if (!Ty) continue;

    // Build the path using the argument index and the total byte offset.
    SmallVector<unsigned, 4> MemberPath;
    MemberPath.push_back(A.getArgNo());
    MemberPath.push_back(TotalOffset);

    // If it's a struct, recurse with the new total offset.
    if (Ty->isStructTy()) {
      traversePointerMembers(A, GEP, TotalOffset, F, AM, ScalarWeights, PointerWeights);
      continue;
    }

    // Analyze scalar members.
    if (Ty->isIntegerTy() || Ty->isFloatingPointTy()) {
      for (User *U2 : GEP->users()) {
        if (auto *LI = dyn_cast<LoadInst>(U2)) {
          if (LI->getType()->isPointerTy()) {
            int W = analyzePointerValue(LI, F, AM);
            if (W > 0) {
              ParameterInfo P(A.getName().str(), &A, MemberPath);
              auto It = PointerWeights.find(P);
              if (It == PointerWeights.end() || It->second < W)
                PointerWeights[P] = W;
              LLVM_DEBUG({
                dbgs() << "  Struct pointer member offset=" << TotalOffset;
                dbgs() << ": weight = " << W << "\n";
              });
            }
            // Recurse on the loaded pointer for pointers to structs.
            traversePointerMembers(A, LI, TotalOffset, F, AM, ScalarWeights, PointerWeights);
          } else {
            int W = analyzeScalarValue(LI, F, AM);
            if (W > 0) {
              ParameterInfo P(A.getName().str(), &A, MemberPath);
              auto It = ScalarWeights.find(P);
              if (It == ScalarWeights.end() || It->second < W)
                ScalarWeights[P] = W;
              LLVM_DEBUG({
                dbgs() << "  Struct scalar member offset=" << TotalOffset;
                dbgs() << ": weight = " << W << "\n";
              });
            }
          }
        }
      }
      continue;
    }

    // Analyze pointer members.
    if (Ty->isPointerTy()) {
      for (User *U2 : GEP->users()) {
        if (auto *LI = dyn_cast<LoadInst>(U2)) {
          int W = analyzePointerValue(LI, F, AM);
          if (W > 0) {
            ParameterInfo P(A.getName().str(), &A, MemberPath);
            auto It = PointerWeights.find(P);
            if (It == PointerWeights.end() || It->second < W)
              PointerWeights[P] = W;
            LLVM_DEBUG({
              dbgs() << "  Struct pointer member offset=" << TotalOffset;
              dbgs() << ": weight = " << W << "\n";
            });
          }
          // Also recurse for pointers to structs.
          traversePointerMembers(A, LI, TotalOffset, F, AM, ScalarWeights, PointerWeights);
        }
      }
      continue;
    }
  }
}

// Generic function to analyze scalar values (arguments or extracted values)
int analyzeScalarValue(const Value *V, Function &F, FunctionAnalysisManager &AM) {
  if (!V->getType()->isIntegerTy() && !V->getType()->isFloatingPointTy()) {
    return 0;
  }

  int weight = 1;
  auto &LI = AM.getResult<LoopAnalysis>(F);
  for (const User *U : V->users()) {
    if (const Instruction *I = dyn_cast<Instruction>(U)) {
      // 控制流直接影响
      if (isa<BranchInst>(I) || isa<SwitchInst>(I)) {
        weight += 10;
      } else if (isa<CmpInst>(I)) {
        for (const User *CmpUser : I->users()) {
          if (const Instruction *CmpI = dyn_cast<Instruction>(CmpUser)) {
            if (isa<BranchInst>(CmpI) || isa<SwitchInst>(CmpI)) {
              weight += 5;
            }
          }
        }
      }

      // 循环内使用（如作为乘数/加数等参与计算）
      if (LI.getLoopFor(I->getParent())) {
        weight += 3;
      }

      // 作为存储值
      if (const StoreInst *SI = dyn_cast<StoreInst>(I)) {
        if (SI->getValueOperand() == V) {
          weight += 4;
        }
      }

      // 参与算术/位运算
      if (isa<BinaryOperator>(I)) {
        weight += 2;
      }
    }
  }
  return weight;
}

// Generic function to analyze pointer values (arguments or extracted values)
int analyzePointerValue(const Value *V, Function &F, FunctionAnalysisManager &AM) {

  //TSET
  //for now give a initial none zero weight, to find more possible opt chance.
  int weight = 1;
  auto &LI = AM.getResult<LoopAnalysis>(F);
  
  for (const User *U : V->users()) {
    if (const Instruction *I = dyn_cast<Instruction>(U)) {

      // if used inside a loop
      if (LI.getLoopFor(I->getParent())) {
        weight += 5;
      }
      
      if (isa<LoadInst>(I) || isa<StoreInst>(I)) {
        weight += 3; // Memory operations benefit more from noalias
      } else if (isa<GetElementPtrInst>(I)) {
        weight += 2; // Pointer arithmetic operations
      }
      
      //if used in loop-invariant ways
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
                         const SmallVector<unsigned, 4> &ParentIndices) {
  Type *BaseTy = A.getType();
  bool IsByVal = A.hasByValAttr();
  if (IsByVal) {
    BaseTy = A.getParamByValType();
  }

  if (!BaseTy->isStructTy()) {
    return;
  }

  // For by-value structs, we trace GEPs from the argument. For direct structs,
  // we would trace ExtractValue instructions.
  if (IsByVal) {
    traversePointerMembers(A, const_cast<Argument *>(&A), 0, F, AM, ScalarWeights, PointerWeights);
    return;
  }

  StructType *ST = cast<StructType>(BaseTy);
  const DataLayout &DL = F.getParent()->getDataLayout();
  const StructLayout *SL = DL.getStructLayout(ST);

  // Analyze each struct member
  for (unsigned i = 0; i < ST->getNumElements(); ++i) {
    Type *MemberType = ST->getElementType(i);
    uint64_t MemberOffset = SL->getElementOffset(i);
    
    // Create indices for this member using offset
    SmallVector<unsigned, 4> MemberPath;
    MemberPath.push_back(ArgIndex);
    MemberPath.push_back(MemberOffset);
    
    // Look for extractvalue instructions that access this member
    for (const User *U : A.users()) {
      if (const ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(U)) {
        // Check if EVI indices match the logical index i
        if (EVI->getNumIndices() == 1 && EVI->getIndices()[0] == i) {
          // This extractvalue accesses our member
          
          // Analyze scalar members
          if (MemberType->isIntegerTy() || MemberType->isFloatingPointTy()) {
            int weight = analyzeScalarValue(EVI, F, AM);
            if (weight > 0) {
              ParameterInfo ParamInfo(A.getName().str(), &A, MemberPath);
              ScalarWeights[ParamInfo] = weight;
            }
          }
          
          // Analyze pointer members
          if (MemberType->isPointerTy()) {
            int weight = analyzePointerValue(EVI, F, AM);
            if (weight > 0) {
              ParameterInfo ParamInfo(A.getName().str(), &A, MemberPath);
              PointerWeights[ParamInfo] = weight;
            }
          }
          
          // Recursively analyze nested struct members
          if (MemberType->isStructTy()) {
            // TODO: implement this
            // This path is not fully implemented for EVI, as it requires
            // more complex offset tracking. The GEP path is preferred.
          }
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
  // TODO: for now only analyze the argument from the function signature, 
  // but we should also analyze hiding scalar like the grid and block dims, and shared memory size maybe.
  for (const Argument &A : F.args()) {
    if (A.hasByValAttr()) {
      // By-value struct arguments are pointers, but we analyze their members.
      analyzeStructMembers(A, ArgIndex, F, AM, ScalarWeights, PointerWeights);
    } else if (A.getType()->isPointerTy()) {
      int weight = analyzePointerValue(&A, F, AM);
      SmallVector<unsigned, 4> TopLevelPath;
      TopLevelPath.push_back(ArgIndex);
      TopLevelPath.push_back(0);
      ParameterInfo ParamInfo(A.getName().str(), &A, TopLevelPath);
      PointerWeights[ParamInfo] = weight;
      LLVM_DEBUG(dbgs() << "  Pointer argument '" << A.getName() 
                       << "': weight = " << weight << "\n");
      // 指向结构体的指针：沿GEP链分析其成员
      traversePointerMembers(A, const_cast<Argument*>(&A), 0, F, AM,
                             ScalarWeights, PointerWeights);
    } else if (A.getType()->isIntegerTy() || A.getType()->isFloatingPointTy()) {
      int weight = analyzeScalarValue(&A, F, AM);
      SmallVector<unsigned, 4> TopLevelPath;
      TopLevelPath.push_back(ArgIndex);
      TopLevelPath.push_back(0);
      ParameterInfo ParamInfo(A.getName().str(), &A, TopLevelPath);
      ScalarWeights[ParamInfo] = weight;
      LLVM_DEBUG(dbgs() << "  Scalar argument '" << A.getName() 
                       << "': weight = " << weight << "\n");
    } else if (A.getType()->isStructTy()) {
      // Analyze struct members
      analyzeStructMembers(A, ArgIndex, F, AM, ScalarWeights, PointerWeights);
      LLVM_DEBUG(dbgs() << "  Struct argument '" << A.getName() 
                       << ": analyzing " << cast<StructType>(A.getType())->getNumElements() 
                       << " members\n");
    }
    ArgIndex++;
  }

  // Analyze uses of special CUDA variables (blockDim, gridDim)
  for (Instruction &I : instructions(F)) {
    if (auto *CI = dyn_cast<CallInst>(&I)) {
      Function *CalledF = CI->getCalledFunction();
      if (CalledF) {
        LLVM_DEBUG(dbgs() << "[CudaKernelAnalysis] Analyzing call: " << CalledF->getName() << "\n");
        StringRef Name = CalledF->getName();
        unsigned SpecialIndex = 0;
        const char* SpecialName = nullptr;

        if (Name == "llvm.nvvm.read.ptx.sreg.nctaid.x") {
          SpecialIndex = GRID_DIM_X_INDEX;
          SpecialName = "gridDim.x";
        } else if (Name == "llvm.nvvm.read.ptx.sreg.nctaid.y") {
          SpecialIndex = GRID_DIM_Y_INDEX;
          SpecialName = "gridDim.y";
        } else if (Name == "llvm.nvvm.read.ptx.sreg.nctaid.z") {
          SpecialIndex = GRID_DIM_Z_INDEX;
          SpecialName = "gridDim.z";
        } else if (Name == "llvm.nvvm.read.ptx.sreg.ntid.x") {
          SpecialIndex = BLOCK_DIM_X_INDEX;
          SpecialName = "blockDim.x";
        } else if (Name == "llvm.nvvm.read.ptx.sreg.ntid.y") {
          SpecialIndex = BLOCK_DIM_Y_INDEX;
          SpecialName = "blockDim.y";
        } else if (Name == "llvm.nvvm.read.ptx.sreg.ntid.z") {
          SpecialIndex = BLOCK_DIM_Z_INDEX;
          SpecialName = "blockDim.z";
        }

        if (SpecialIndex != 0) {
          int weight = analyzeScalarValue(CI, F, AM);
          LLVM_DEBUG(dbgs() << "[CudaKernelAnalysis] Found special intrinsic '" << Name
                            << "' with calculated weight " << weight << "\n");
          if (weight > 0) {
            SmallVector<unsigned, 4> Path;
            Path.push_back(SpecialIndex);
            Path.push_back(0);
            // The name in ParameterInfo is mainly for debug printing.
            // The Argument* is null, which is fine as long as we don't access it without checking.
            ParameterInfo PI(SpecialName, nullptr, Path);
            
            // The same intrinsic can be called multiple times. We want to calculate
            // the weight based on all its uses, but `analyzeScalarValue` does that
            // for one call site. We should aggregate weights from all call sites
            // for the same implicit parameter.
            ScalarWeights[PI] += weight;
          }
        }
      } 
    }
  }

  // 汇总打印：将结构体成员并入 Scalar/Pointer 列表
  LLVM_DEBUG({
    dbgs() << "  Scalar arguments (including struct members):\n";
    for (const auto &KV : ScalarWeights) {
      const ParameterInfo &P = KV.first; int W = KV.second;
      dbgs() << "    - indexPath=";
      for (size_t i = 0; i < P.Indices.size(); ++i) dbgs() << (i? "." : "") << P.Indices[i];
      dbgs() << " name='" << P.getDisplayName() << "' weight=" << W << "\n";
    }
    dbgs() << "  Pointer arguments (including struct members):\n";
    for (const auto &KV : PointerWeights) {
      const ParameterInfo &P = KV.first; int W = KV.second;
      dbgs() << "    - indexPath=";
      for (size_t i = 0; i < P.Indices.size(); ++i) dbgs() << (i? "." : "") << P.Indices[i];
      dbgs() << " name='" << P.getDisplayName() << "' weight=" << W << "\n";
    }
  });

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