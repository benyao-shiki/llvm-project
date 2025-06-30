#include "llvm/Analysis/InterpAliasAnalysis.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/PassRegistry.h"
#include "llvm/Pass.h"

using namespace llvm;

// Initialize the analysis key
AnalysisKey InterpAA::Key;

// Anderson Analysis Implementation
AndersonAnalysis::AndersonAnalysis(Module &M) {
  buildPointsToSets(M);
}

const std::unordered_set<Value*>& AndersonAnalysis::getPointsToSet(Value *V) const {
  auto it = PointsToMap.find(V);
  if (it != PointsToMap.end()) {
    return it->second;
  }
  static const std::unordered_set<Value*> EmptySet;
  return EmptySet;
}

void AndersonAnalysis::buildPointsToSets(Module &M) {
  // Initialize points-to sets
  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (AllocaInst *AI = dyn_cast<AllocaInst>(&I)) {
          PointsToMap[AI].insert(AI);
        }
        else if (CallBase *CB = dyn_cast<CallBase>(&I)) {
          // Handle cudaMalloc calls
          if (CB->getCalledFunction() && 
              CB->getCalledFunction()->getName() == "cudaMalloc") {
            if (CB->arg_size() >= 1) {
              Value *PtrArg = CB->getArgOperand(0);
              if (PtrArg) {
                // Each cudaMalloc call allocates a unique memory location
                PointsToMap[PtrArg].insert(CB);
              }
            }
          }
        }
      }
    }
  }
}

// Run the analysis on a module
InterpAAResult InterpAA::run(Module &M, ModuleAnalysisManager &AM) {
  return InterpAAResult(M);
}

// Constructor for the analysis result
InterpAAResult::InterpAAResult(Module &M) : Analysis(std::make_unique<AndersonAnalysis>(M)) {
}

// Move constructor
InterpAAResult::InterpAAResult(InterpAAResult &&Arg) noexcept 
  : AAResultBase(std::move(Arg)),
    Analysis(std::move(Arg.Analysis)) {
}

// Destructor
InterpAAResult::~InterpAAResult() = default;

// Determine if two memory locations can alias
AliasResult InterpAAResult::alias(const MemoryLocation &LocA,
                                 const MemoryLocation &LocB,
                                 AAQueryInfo &AAQI,
                                 const Instruction *CtxI) {
  Value *PtrA = const_cast<Value*>(LocA.Ptr);
  Value *PtrB = const_cast<Value*>(LocB.Ptr);
  
  const auto &PointsToA = Analysis->getPointsToSet(PtrA);
  const auto &PointsToB = Analysis->getPointsToSet(PtrB);

  // If either points-to set is empty, conservatively return MayAlias
  if (PointsToA.empty() || PointsToB.empty())
    return AliasResult::MayAlias;

  // Check if the points-to sets intersect
  for (Value *VA : PointsToA) {
    if (PointsToB.count(VA) > 0)
      return AliasResult::MustAlias;
  }

  return AliasResult::NoAlias;
}

// Get mod/ref behavior of a call site
ModRefInfo InterpAAResult::getModRefInfo(const CallBase *Call,
                                        const MemoryLocation &Loc,
                                        AAQueryInfo &AAQI) {
  // For now, return ModRef as a conservative default
  return ModRefInfo::ModRef;
}

// Get mod/ref behavior between two call sites
ModRefInfo InterpAAResult::getModRefInfo(const CallBase *Call1,
                                        const CallBase *Call2,
                                        AAQueryInfo &AAQI) {
  // For now, return ModRef as a conservative default
  return ModRefInfo::ModRef;
}

// Register the pass
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "InterpAA", "v0.1",
    [](PassBuilder &PB) {
      // Register the pass with the new PM
      PB.registerAnalysisRegistrationCallback(
        [](ModuleAnalysisManager &MAM) {
          MAM.registerPass([&] { return InterpAA(); });
        });
      
      // Register the pass as an alias analysis
      PB.registerPipelineParsingCallback(
        [](StringRef Name, ModulePassManager &MPM,
           ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "interpaa") {
            MPM.addPass(RequireAnalysisPass<InterpAA, Module>());
            return true;
          }
          return false;
        });
    }
  };
} 