//===- InterpAliasAnalysis.h - Interprocedural Alias Analysis ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the InterpAA analysis pass, which implements
/// interprocedural alias analysis based on Anderson's algorithm.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_INTERPALIASANALYSIS_H
#define LLVM_ANALYSIS_INTERPALIASANALYSIS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Pass.h"
#include <unordered_map>
#include <unordered_set>

namespace llvm {

class Module;
class Instruction;
class Value;
class Function;
class raw_ostream;
class CallBase;
class PassRegistry;

// Forward declaration of the Anderson analysis implementation
class AndersonAnalysis;

void initializeInterpAAPass(PassRegistry &Registry);

class AndersonAnalysis {
public:
  AndersonAnalysis(Module &M);
  ~AndersonAnalysis() = default;

  // Get the points-to set for a value
  const std::unordered_set<Value*>& getPointsToSet(Value *V) const;

private:
  // Map from values to their points-to sets
  std::unordered_map<Value*, std::unordered_set<Value*>> PointsToMap;

  // Build points-to sets for the module
  void buildPointsToSets(Module &M);
};

// InterpAA Result class - Implements the interprocedural alias analysis
class InterpAAResult : public AAResultBase {
  std::unique_ptr<AndersonAnalysis> Analysis;

public:
  explicit InterpAAResult(Module &M);
  InterpAAResult(InterpAAResult &&Arg) noexcept;
  ~InterpAAResult();

  AliasResult alias(const MemoryLocation &LocA, const MemoryLocation &LocB,
                    AAQueryInfo &AAQI, const Instruction *CtxI = nullptr);

  ModRefInfo getModRefInfo(const CallBase *Call, const MemoryLocation &Loc,
                         AAQueryInfo &AAQI);

  ModRefInfo getModRefInfo(const CallBase *Call1, const CallBase *Call2,
                           AAQueryInfo &AAQI);
};

/// Analysis pass that implements interprocedural alias analysis
class InterpAA : public AnalysisInfoMixin<InterpAA> {
  friend AnalysisInfoMixin<InterpAA>;
public:
  static AnalysisKey Key;

  using Result = InterpAAResult;
  
  static StringRef name() { return "InterpAA"; }
  
  InterpAAResult run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_ANALYSIS_INTERPALIASANALYSIS_H