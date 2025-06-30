//===-- Analysis.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Analysis.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/InterpAliasAnalysis.h"
#include <cstring>

using namespace llvm;

/// initializeAnalysis - Initialize all passes linked into the Analysis library.
void llvm::initializeAnalysis(PassRegistry &Registry) {
  initializeAAResultsWrapperPassPass(Registry);
  initializeBasicAAWrapperPassPass(Registry);
  initializeGlobalsAAWrapperPassPass(Registry);
  initializeTypeBasedAAWrapperPassPass(Registry);
  initializeInterpAAWrapperPassPass(Registry);
}

void llvm::initializeInterpAAWrapperPassPass(PassRegistry &Registry) {
  PassInfo *PI = new PassInfo("Interprocedural Alias Analysis", "interpaa",
                             &InterpAA::Key, nullptr, true, true);
  Registry.registerPass(*PI, true);
}

LLVMBool LLVMVerifyModule(LLVMModuleRef M, LLVMVerifierFailureAction Action,
                          char **OutMessages) {
  raw_ostream *DebugOS = Action != LLVMReturnStatusAction ? &errs() : nullptr;
  std::string Messages;
  raw_string_ostream MsgsOS(Messages);

  LLVMBool Result = verifyModule(*unwrap(M), OutMessages ? &MsgsOS : DebugOS);

  // Duplicate the output to stderr.
  if (DebugOS && OutMessages)
    *DebugOS << MsgsOS.str();

  if (Action == LLVMAbortProcessAction && Result)
    report_fatal_error("Broken module found, compilation aborted!");

  if (OutMessages)
    *OutMessages = strdup(MsgsOS.str().c_str());

  return Result;
}

LLVMBool LLVMVerifyFunction(LLVMValueRef Fn, LLVMVerifierFailureAction Action) {
  LLVMBool Result = verifyFunction(
      *unwrap<Function>(Fn), Action != LLVMReturnStatusAction ? &errs()
                                                              : nullptr);

  if (Action == LLVMAbortProcessAction && Result)
    report_fatal_error("Broken function found, compilation aborted!");

  return Result;
}

void LLVMViewFunctionCFG(LLVMValueRef Fn) {
  Function *F = unwrap<Function>(Fn);
  F->viewCFG();
}

void LLVMViewFunctionCFGOnly(LLVMValueRef Fn) {
  Function *F = unwrap<Function>(Fn);
  F->viewCFGOnly();
}
