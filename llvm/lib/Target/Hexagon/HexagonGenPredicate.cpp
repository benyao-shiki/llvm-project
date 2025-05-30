//===- HexagonGenPredicate.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Hexagon.h"
#include "HexagonInstrInfo.h"
#include "HexagonSubtarget.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <iterator>
#include <queue>
#include <utility>

#define DEBUG_TYPE "gen-pred"

using namespace llvm;

namespace {

using RegSubRegPair = TargetInstrInfo::RegSubRegPair;

struct PrintRegister {
  friend raw_ostream &operator<<(raw_ostream &OS, const PrintRegister &PR);

  PrintRegister(RegSubRegPair R, const TargetRegisterInfo &I)
      : Reg(R), TRI(I) {}

private:
  RegSubRegPair Reg;
  const TargetRegisterInfo &TRI;
};

  raw_ostream &operator<< (raw_ostream &OS, const PrintRegister &PR)
    LLVM_ATTRIBUTE_UNUSED;
  raw_ostream &operator<< (raw_ostream &OS, const PrintRegister &PR) {
    return OS << printReg(PR.Reg.Reg, &PR.TRI, PR.Reg.SubReg);
  }

  class HexagonGenPredicate : public MachineFunctionPass {
  public:
    static char ID;

    HexagonGenPredicate() : MachineFunctionPass(ID) {}

    StringRef getPassName() const override {
      return "Hexagon generate predicate operations";
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<MachineDominatorTreeWrapperPass>();
      AU.addPreserved<MachineDominatorTreeWrapperPass>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    bool runOnMachineFunction(MachineFunction &MF) override;

  private:
    using VectOfInst = SetVector<MachineInstr *>;
    using SetOfReg = SetVector<RegSubRegPair>;
    using RegToRegMap = DenseMap<RegSubRegPair, RegSubRegPair>;

    const HexagonInstrInfo *TII = nullptr;
    const HexagonRegisterInfo *TRI = nullptr;
    MachineRegisterInfo *MRI = nullptr;
    SetOfReg PredGPRs;
    VectOfInst PUsers;
    RegToRegMap G2P;

    bool isPredReg(Register R);
    void collectPredicateGPR(MachineFunction &MF);
    void processPredicateGPR(const RegSubRegPair &Reg);
    unsigned getPredForm(unsigned Opc);
    bool isConvertibleToPredForm(const MachineInstr *MI);
    bool isScalarCmp(unsigned Opc);
    bool isScalarPred(RegSubRegPair PredReg);
    RegSubRegPair getPredRegFor(const RegSubRegPair &Reg);
    bool convertToPredForm(MachineInstr *MI);
    bool eliminatePredCopies(MachineFunction &MF);
  };

} // end anonymous namespace

char HexagonGenPredicate::ID = 0;

INITIALIZE_PASS_BEGIN(HexagonGenPredicate, "hexagon-gen-pred-pass",
                      "Hexagon generate predicate operations", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(HexagonGenPredicate, "hexagon-gen-pred-pass",
                    "Hexagon generate predicate operations", false, false)

bool HexagonGenPredicate::isPredReg(Register R) {
  if (!R.isVirtual())
    return false;
  const TargetRegisterClass *RC = MRI->getRegClass(R);
  return RC == &Hexagon::PredRegsRegClass;
}

unsigned HexagonGenPredicate::getPredForm(unsigned Opc) {
  using namespace Hexagon;

  switch (Opc) {
    case A2_and:
    case A2_andp:
      return C2_and;
    case A4_andn:
    case A4_andnp:
      return C2_andn;
    case M4_and_and:
      return C4_and_and;
    case M4_and_andn:
      return C4_and_andn;
    case M4_and_or:
      return C4_and_or;

    case A2_or:
    case A2_orp:
      return C2_or;
    case A4_orn:
    case A4_ornp:
      return C2_orn;
    case M4_or_and:
      return C4_or_and;
    case M4_or_andn:
      return C4_or_andn;
    case M4_or_or:
      return C4_or_or;

    case A2_xor:
    case A2_xorp:
      return C2_xor;

    case C2_tfrrp:
      return COPY;
  }
  // The opcode corresponding to 0 is TargetOpcode::PHI. We can use 0 here
  // to denote "none", but we need to make sure that none of the valid opcodes
  // that we return will ever be 0.
  static_assert(PHI == 0, "Use different value for <none>");
  return 0;
}

bool HexagonGenPredicate::isConvertibleToPredForm(const MachineInstr *MI) {
  unsigned Opc = MI->getOpcode();
  if (getPredForm(Opc) != 0)
    return true;

  // Comparisons against 0 are also convertible. This does not apply to
  // A4_rcmpeqi or A4_rcmpneqi, since they produce values 0 or 1, which
  // may not match the value that the predicate register would have if
  // it was converted to a predicate form.
  switch (Opc) {
    case Hexagon::C2_cmpeqi:
    case Hexagon::C4_cmpneqi:
      if (MI->getOperand(2).isImm() && MI->getOperand(2).getImm() == 0)
        return true;
      break;
  }
  return false;
}

void HexagonGenPredicate::collectPredicateGPR(MachineFunction &MF) {
  for (MachineBasicBlock &B : MF) {
    for (MachineInstr &MI : B) {
      unsigned Opc = MI.getOpcode();
      switch (Opc) {
        case Hexagon::C2_tfrpr:
        case TargetOpcode::COPY:
          if (isPredReg(MI.getOperand(1).getReg())) {
            RegSubRegPair RD = getRegSubRegPair(MI.getOperand(0));
            if (RD.Reg.isVirtual())
              PredGPRs.insert(RD);
          }
          break;
      }
    }
  }
}

void HexagonGenPredicate::processPredicateGPR(const RegSubRegPair &Reg) {
  LLVM_DEBUG(dbgs() << __func__ << ": " << printReg(Reg.Reg, TRI, Reg.SubReg)
                    << "\n");
  using use_iterator = MachineRegisterInfo::use_iterator;

  use_iterator I = MRI->use_begin(Reg.Reg), E = MRI->use_end();
  if (I == E) {
    LLVM_DEBUG(dbgs() << "Dead reg: " << printReg(Reg.Reg, TRI, Reg.SubReg)
                      << '\n');
    MachineInstr *DefI = MRI->getVRegDef(Reg.Reg);
    DefI->eraseFromParent();
    return;
  }

  for (; I != E; ++I) {
    MachineInstr *UseI = I->getParent();
    if (isConvertibleToPredForm(UseI))
      PUsers.insert(UseI);
  }
}

RegSubRegPair HexagonGenPredicate::getPredRegFor(const RegSubRegPair &Reg) {
  // Create a predicate register for a given Reg. The newly created register
  // will have its value copied from Reg, so that it can be later used as
  // an operand in other instructions.
  assert(Reg.Reg.isVirtual());
  RegToRegMap::iterator F = G2P.find(Reg);
  if (F != G2P.end())
    return F->second;

  LLVM_DEBUG(dbgs() << __func__ << ": " << PrintRegister(Reg, *TRI));
  MachineInstr *DefI = MRI->getVRegDef(Reg.Reg);
  assert(DefI);
  unsigned Opc = DefI->getOpcode();
  if (Opc == Hexagon::C2_tfrpr || Opc == TargetOpcode::COPY) {
    assert(DefI->getOperand(0).isDef() && DefI->getOperand(1).isUse());
    RegSubRegPair PR = getRegSubRegPair(DefI->getOperand(1));
    G2P.insert(std::make_pair(Reg, PR));
    LLVM_DEBUG(dbgs() << " -> " << PrintRegister(PR, *TRI) << '\n');
    return PR;
  }

  MachineBasicBlock &B = *DefI->getParent();
  DebugLoc DL = DefI->getDebugLoc();
  const TargetRegisterClass *PredRC = &Hexagon::PredRegsRegClass;
  Register NewPR = MRI->createVirtualRegister(PredRC);

  // For convertible instructions, do not modify them, so that they can
  // be converted later.  Generate a copy from Reg to NewPR.
  if (isConvertibleToPredForm(DefI)) {
    MachineBasicBlock::iterator DefIt = DefI;
    BuildMI(B, std::next(DefIt), DL, TII->get(TargetOpcode::COPY), NewPR)
        .addReg(Reg.Reg, 0, Reg.SubReg);
    G2P.insert(std::make_pair(Reg, RegSubRegPair(NewPR)));
    LLVM_DEBUG(dbgs() << " -> !" << PrintRegister(RegSubRegPair(NewPR), *TRI)
                      << '\n');
    return RegSubRegPair(NewPR);
  }

  llvm_unreachable("Invalid argument");
}

bool HexagonGenPredicate::isScalarCmp(unsigned Opc) {
  switch (Opc) {
    case Hexagon::C2_cmpeq:
    case Hexagon::C2_cmpgt:
    case Hexagon::C2_cmpgtu:
    case Hexagon::C2_cmpeqp:
    case Hexagon::C2_cmpgtp:
    case Hexagon::C2_cmpgtup:
    case Hexagon::C2_cmpeqi:
    case Hexagon::C2_cmpgti:
    case Hexagon::C2_cmpgtui:
    case Hexagon::C2_cmpgei:
    case Hexagon::C2_cmpgeui:
    case Hexagon::C4_cmpneqi:
    case Hexagon::C4_cmpltei:
    case Hexagon::C4_cmplteui:
    case Hexagon::C4_cmpneq:
    case Hexagon::C4_cmplte:
    case Hexagon::C4_cmplteu:
    case Hexagon::A4_cmpbeq:
    case Hexagon::A4_cmpbeqi:
    case Hexagon::A4_cmpbgtu:
    case Hexagon::A4_cmpbgtui:
    case Hexagon::A4_cmpbgt:
    case Hexagon::A4_cmpbgti:
    case Hexagon::A4_cmpheq:
    case Hexagon::A4_cmphgt:
    case Hexagon::A4_cmphgtu:
    case Hexagon::A4_cmpheqi:
    case Hexagon::A4_cmphgti:
    case Hexagon::A4_cmphgtui:
      return true;
  }
  return false;
}

bool HexagonGenPredicate::isScalarPred(RegSubRegPair PredReg) {
  std::queue<RegSubRegPair> WorkQ;
  WorkQ.push(PredReg);

  while (!WorkQ.empty()) {
    RegSubRegPair PR = WorkQ.front();
    WorkQ.pop();
    const MachineInstr *DefI = MRI->getVRegDef(PR.Reg);
    if (!DefI)
      return false;
    unsigned DefOpc = DefI->getOpcode();
    switch (DefOpc) {
      case TargetOpcode::COPY: {
        const TargetRegisterClass *PredRC = &Hexagon::PredRegsRegClass;
        if (MRI->getRegClass(PR.Reg) != PredRC)
          return false;
        // If it is a copy between two predicate registers, fall through.
        [[fallthrough]];
      }
      case Hexagon::C2_and:
      case Hexagon::C2_andn:
      case Hexagon::C4_and_and:
      case Hexagon::C4_and_andn:
      case Hexagon::C4_and_or:
      case Hexagon::C2_or:
      case Hexagon::C2_orn:
      case Hexagon::C4_or_and:
      case Hexagon::C4_or_andn:
      case Hexagon::C4_or_or:
      case Hexagon::C4_or_orn:
      case Hexagon::C2_xor:
        // Add operands to the queue.
        for (const MachineOperand &MO : DefI->operands())
          if (MO.isReg() && MO.isUse())
            WorkQ.push(RegSubRegPair(MO.getReg()));
        break;

      // All non-vector compares are ok, everything else is bad.
      default:
        return isScalarCmp(DefOpc);
    }
  }

  return true;
}

bool HexagonGenPredicate::convertToPredForm(MachineInstr *MI) {
  LLVM_DEBUG(dbgs() << __func__ << ": " << MI << " " << *MI);

  unsigned Opc = MI->getOpcode();
  assert(isConvertibleToPredForm(MI));
  unsigned NumOps = MI->getNumOperands();
  for (unsigned i = 0; i < NumOps; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isUse())
      continue;
    RegSubRegPair Reg(getRegSubRegPair(MO));
    if (Reg.SubReg && Reg.SubReg != Hexagon::isub_lo)
      return false;
    if (!PredGPRs.count(Reg))
      return false;
  }

  MachineBasicBlock &B = *MI->getParent();
  DebugLoc DL = MI->getDebugLoc();

  unsigned NewOpc = getPredForm(Opc);
  // Special case for comparisons against 0.
  if (NewOpc == 0) {
    switch (Opc) {
      case Hexagon::C2_cmpeqi:
        NewOpc = Hexagon::C2_not;
        break;
      case Hexagon::C4_cmpneqi:
        NewOpc = TargetOpcode::COPY;
        break;
      default:
        return false;
    }

    // If it's a scalar predicate register, then all bits in it are
    // the same. Otherwise, to determine whether all bits are 0 or not
    // we would need to use any8.
    RegSubRegPair PR = getPredRegFor(getRegSubRegPair(MI->getOperand(1)));
    if (!isScalarPred(PR))
      return false;
    // This will skip the immediate argument when creating the predicate
    // version instruction.
    NumOps = 2;
  }

  // Check that def is in operand #0.
  MachineOperand &Op0 = MI->getOperand(0);
  assert(Op0.isDef());
  RegSubRegPair OutR(getRegSubRegPair(Op0));

  // Don't use getPredRegFor, since it will create an association between
  // the argument and a created predicate register (i.e. it will insert a
  // copy if a new predicate register is created).
  const TargetRegisterClass *PredRC = &Hexagon::PredRegsRegClass;
  RegSubRegPair NewPR = MRI->createVirtualRegister(PredRC);
  MachineInstrBuilder MIB = BuildMI(B, MI, DL, TII->get(NewOpc), NewPR.Reg);

  // Add predicate counterparts of the GPRs.
  for (unsigned i = 1; i < NumOps; ++i) {
    RegSubRegPair GPR = getRegSubRegPair(MI->getOperand(i));
    RegSubRegPair Pred = getPredRegFor(GPR);
    MIB.addReg(Pred.Reg, 0, Pred.SubReg);
  }
  LLVM_DEBUG(dbgs() << "generated: " << *MIB);

  // Generate a copy-out: NewGPR = NewPR, and replace all uses of OutR
  // with NewGPR.
  const TargetRegisterClass *RC = MRI->getRegClass(OutR.Reg);
  Register NewOutR = MRI->createVirtualRegister(RC);
  BuildMI(B, MI, DL, TII->get(TargetOpcode::COPY), NewOutR)
      .addReg(NewPR.Reg, 0, NewPR.SubReg);
  MRI->replaceRegWith(OutR.Reg, NewOutR);
  MI->eraseFromParent();

  // If the processed instruction was C2_tfrrp (i.e. Rn = Pm; Pk = Rn),
  // then the output will be a predicate register.  Do not visit the
  // users of it.
  if (!isPredReg(NewOutR)) {
    RegSubRegPair R(NewOutR);
    PredGPRs.insert(R);
    processPredicateGPR(R);
  }
  return true;
}

bool HexagonGenPredicate::eliminatePredCopies(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << __func__ << "\n");
  const TargetRegisterClass *PredRC = &Hexagon::PredRegsRegClass;
  bool Changed = false;
  VectOfInst Erase;

  // First, replace copies
  //   IntR = PredR1
  //   PredR2 = IntR
  // with
  //   PredR2 = PredR1
  // Such sequences can be generated when a copy-into-pred is generated from
  // a gpr register holding a result of a convertible instruction. After
  // the convertible instruction is converted, its predicate result will be
  // copied back into the original gpr.

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() != TargetOpcode::COPY)
        continue;
      RegSubRegPair DR = getRegSubRegPair(MI.getOperand(0));
      RegSubRegPair SR = getRegSubRegPair(MI.getOperand(1));
      if (!DR.Reg.isVirtual())
        continue;
      if (!SR.Reg.isVirtual())
        continue;
      if (MRI->getRegClass(DR.Reg) != PredRC)
        continue;
      if (MRI->getRegClass(SR.Reg) != PredRC)
        continue;
      assert(!DR.SubReg && !SR.SubReg && "Unexpected subregister");
      MRI->replaceRegWith(DR.Reg, SR.Reg);
      Erase.insert(&MI);
      Changed = true;
    }
  }

  for (MachineInstr *MI : Erase)
    MI->eraseFromParent();

  return Changed;
}

bool HexagonGenPredicate::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  TII = MF.getSubtarget<HexagonSubtarget>().getInstrInfo();
  TRI = MF.getSubtarget<HexagonSubtarget>().getRegisterInfo();
  MRI = &MF.getRegInfo();
  PredGPRs.clear();
  PUsers.clear();
  G2P.clear();

  bool Changed = false;
  collectPredicateGPR(MF);
  for (const RegSubRegPair &R : PredGPRs)
    processPredicateGPR(R);

  bool Again;
  do {
    Again = false;
    VectOfInst Processed, Copy;

    Copy = PUsers;
    for (MachineInstr *MI : Copy) {
      bool Done = convertToPredForm(MI);
      if (Done) {
        Processed.insert(MI);
        Again = true;
      }
    }
    Changed |= Again;

    auto Done = [Processed] (MachineInstr *MI) -> bool {
      return Processed.count(MI);
    };
    PUsers.remove_if(Done);
  } while (Again);

  Changed |= eliminatePredCopies(MF);
  return Changed;
}

FunctionPass *llvm::createHexagonGenPredicate() {
  return new HexagonGenPredicate();
}
