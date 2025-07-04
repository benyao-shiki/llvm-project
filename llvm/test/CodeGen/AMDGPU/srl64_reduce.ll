; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 5
;; Test reduction of:
;;
;;   DST = lshr i64 X, Y
;;
;; where Y is in the range [63-32] to:
;;
;;   DST = [srl i32 X, (Y & 0x1F), 0]

; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx900 < %s | FileCheck %s

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Test range with metadata
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define i64 @srl_metadata(i64 %arg0, ptr %arg1.ptr) {
; CHECK-LABEL: srl_metadata:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    flat_load_dword v0, v[2:3]
; CHECK-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_lshrrev_b32_e32 v0, v0, v1
; CHECK-NEXT:    v_mov_b32_e32 v1, 0
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %shift.amt = load i64, ptr %arg1.ptr, !range !0, !noundef !{}
  %srl = lshr i64 %arg0, %shift.amt
  ret i64 %srl
}

define amdgpu_ps i64 @srl_metadata_sgpr_return(i64 inreg %arg0, ptr addrspace(1) inreg %arg1.ptr) {
; CHECK-LABEL: srl_metadata_sgpr_return:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_load_dword s0, s[2:3], 0x0
; CHECK-NEXT:    s_waitcnt lgkmcnt(0)
; CHECK-NEXT:    s_lshr_b32 s0, s1, s0
; CHECK-NEXT:    s_mov_b32 s1, 0
; CHECK-NEXT:    ; return to shader part epilog
  %shift.amt = load i64, ptr addrspace(1) %arg1.ptr, !range !0, !noundef !{}
  %srl = lshr i64 %arg0, %shift.amt
  ret i64 %srl
}

; Exact attribute does not inhibit reduction
define i64 @srl_exact_metadata(i64 %arg0, ptr %arg1.ptr) {
; CHECK-LABEL: srl_exact_metadata:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    flat_load_dword v0, v[2:3]
; CHECK-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_lshrrev_b32_e32 v0, v0, v1
; CHECK-NEXT:    v_mov_b32_e32 v1, 0
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %shift.amt = load i64, ptr %arg1.ptr, !range !0, !noundef !{}
  %srl = lshr exact i64 %arg0, %shift.amt
  ret i64 %srl
}

define i64 @srl_metadata_two_ranges(i64 %arg0, ptr %arg1.ptr) {
; CHECK-LABEL: srl_metadata_two_ranges:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    flat_load_dword v0, v[2:3]
; CHECK-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_lshrrev_b32_e32 v0, v0, v1
; CHECK-NEXT:    v_mov_b32_e32 v1, 0
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %shift.amt = load i64, ptr %arg1.ptr, !range !1, !noundef !{}
  %srl = lshr i64 %arg0, %shift.amt
  ret i64 %srl
}

; Known minimum is too low.  Reduction must not be done.
define i64 @srl_metadata_out_of_range(i64 %arg0, ptr %arg1.ptr) {
; CHECK-LABEL: srl_metadata_out_of_range:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    flat_load_dword v2, v[2:3]
; CHECK-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_lshrrev_b64 v[0:1], v2, v[0:1]
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %shift.amt = load i64, ptr %arg1.ptr, !range !2, !noundef !{}
  %srl = lshr i64 %arg0, %shift.amt
  ret i64 %srl
}

; Bounds cannot be truncated to i32 when load is narrowed to i32.
; Reduction must not be done.
; Bounds were chosen so that if bounds were truncated to i32 the
; known minimum would be 32 and the srl would be erroneously reduced.
define i64 @srl_metadata_cant_be_narrowed_to_i32(i64 %arg0, ptr %arg1.ptr) {
; CHECK-LABEL: srl_metadata_cant_be_narrowed_to_i32:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    flat_load_dword v2, v[2:3]
; CHECK-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_lshrrev_b64 v[0:1], v2, v[0:1]
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %shift.amt = load i64, ptr %arg1.ptr, !range !3, !noundef !{}
  %srl = lshr i64 %arg0, %shift.amt
  ret i64 %srl
}

define <2 x i64> @srl_v2_metadata(<2 x i64> %arg0, ptr %arg1.ptr) {
; CHECK-LABEL: srl_v2_metadata:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    flat_load_dwordx4 v[4:7], v[4:5]
; CHECK-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_lshrrev_b32_e32 v0, v4, v1
; CHECK-NEXT:    v_lshrrev_b32_e32 v2, v6, v3
; CHECK-NEXT:    v_mov_b32_e32 v1, 0
; CHECK-NEXT:    v_mov_b32_e32 v3, 0
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %shift.amt = load <2 x i64>, ptr %arg1.ptr, !range !0, !noundef !{}
  %srl = lshr <2 x i64> %arg0, %shift.amt
  ret <2 x i64> %srl
}

; Exact attribute does not inhibit reduction
define <2 x i64> @srl_exact_v2_metadata(<2 x i64> %arg0, ptr %arg1.ptr) {
; CHECK-LABEL: srl_exact_v2_metadata:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    flat_load_dwordx4 v[4:7], v[4:5]
; CHECK-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_lshrrev_b32_e32 v0, v4, v1
; CHECK-NEXT:    v_lshrrev_b32_e32 v2, v6, v3
; CHECK-NEXT:    v_mov_b32_e32 v1, 0
; CHECK-NEXT:    v_mov_b32_e32 v3, 0
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %shift.amt = load <2 x i64>, ptr %arg1.ptr, !range !0, !noundef !{}
  %srl = lshr exact <2 x i64> %arg0, %shift.amt
  ret <2 x i64> %srl
}

define <3 x i64> @srl_v3_metadata(<3 x i64> %arg0, ptr %arg1.ptr) {
; CHECK-LABEL: srl_v3_metadata:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    flat_load_dword v0, v[6:7] offset:16
; CHECK-NEXT:    flat_load_dwordx4 v[8:11], v[6:7]
; CHECK-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_lshrrev_b32_e32 v4, v0, v5
; CHECK-NEXT:    v_lshrrev_b32_e32 v0, v8, v1
; CHECK-NEXT:    v_lshrrev_b32_e32 v2, v10, v3
; CHECK-NEXT:    v_mov_b32_e32 v1, 0
; CHECK-NEXT:    v_mov_b32_e32 v3, 0
; CHECK-NEXT:    v_mov_b32_e32 v5, 0
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %shift.amt = load <3 x i64>, ptr %arg1.ptr, !range !0, !noundef !{}
  %srl = lshr <3 x i64> %arg0, %shift.amt
  ret <3 x i64> %srl
}

define <4 x i64> @srl_v4_metadata(<4 x i64> %arg0, ptr %arg1.ptr) {
; CHECK-LABEL: srl_v4_metadata:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    flat_load_dwordx4 v[10:13], v[8:9]
; CHECK-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; CHECK-NEXT:    flat_load_dwordx4 v[13:16], v[8:9] offset:16
; CHECK-NEXT:    ; kill: killed $vgpr8 killed $vgpr9
; CHECK-NEXT:    v_lshrrev_b32_e32 v0, v10, v1
; CHECK-NEXT:    v_lshrrev_b32_e32 v2, v12, v3
; CHECK-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_lshrrev_b32_e32 v4, v13, v5
; CHECK-NEXT:    v_lshrrev_b32_e32 v6, v15, v7
; CHECK-NEXT:    v_mov_b32_e32 v1, 0
; CHECK-NEXT:    v_mov_b32_e32 v3, 0
; CHECK-NEXT:    v_mov_b32_e32 v5, 0
; CHECK-NEXT:    v_mov_b32_e32 v7, 0
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %shift.amt = load <4 x i64>, ptr %arg1.ptr, !range !0, !noundef !{}
  %srl = lshr <4 x i64> %arg0, %shift.amt
  ret <4 x i64> %srl
}

!0 = !{i64 32, i64 64}
!1 = !{i64 32, i64 38, i64 42, i64 48}
!2 = !{i64 31, i64 38, i64 42, i64 48}
!3 = !{i64 32, i64 38, i64 2147483680, i64 2147483681}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Test range with an "or X, 16"
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; These cases must not be reduced because the known minimum, 16, is not in range.

define i64 @srl_or16(i64 %arg0, i64 %shift_amt) {
; CHECK-LABEL: srl_or16:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_or_b32_e32 v2, 16, v2
; CHECK-NEXT:    v_lshrrev_b64 v[0:1], v2, v[0:1]
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %or = or i64 %shift_amt, 16
  %srl = lshr i64 %arg0, %or
  ret i64 %srl
}

define <2 x i64> @srl_v2_or16(<2 x i64> %arg0, <2 x i64> %shift_amt) {
; CHECK-LABEL: srl_v2_or16:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_or_b32_e32 v5, 16, v6
; CHECK-NEXT:    v_or_b32_e32 v4, 16, v4
; CHECK-NEXT:    v_lshrrev_b64 v[0:1], v4, v[0:1]
; CHECK-NEXT:    v_lshrrev_b64 v[2:3], v5, v[2:3]
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %or = or <2 x i64> %shift_amt, splat (i64 16)
  %srl = lshr <2 x i64> %arg0, %or
  ret <2 x i64> %srl
}

define <3 x i64> @srl_v3_or16(<3 x i64> %arg0, <3 x i64> %shift_amt) {
; CHECK-LABEL: srl_v3_or16:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_or_b32_e32 v7, 16, v10
; CHECK-NEXT:    v_or_b32_e32 v8, 16, v8
; CHECK-NEXT:    v_or_b32_e32 v6, 16, v6
; CHECK-NEXT:    v_lshrrev_b64 v[0:1], v6, v[0:1]
; CHECK-NEXT:    v_lshrrev_b64 v[2:3], v8, v[2:3]
; CHECK-NEXT:    v_lshrrev_b64 v[4:5], v7, v[4:5]
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %or = or <3 x i64> %shift_amt, splat (i64 16)
  %srl = lshr <3 x i64> %arg0, %or
  ret <3 x i64> %srl
}

define <4 x i64> @srl_v4_or16(<4 x i64> %arg0, <4 x i64> %shift_amt) {
; CHECK-LABEL: srl_v4_or16:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_or_b32_e32 v9, 16, v14
; CHECK-NEXT:    v_or_b32_e32 v11, 16, v12
; CHECK-NEXT:    v_or_b32_e32 v10, 16, v10
; CHECK-NEXT:    v_or_b32_e32 v8, 16, v8
; CHECK-NEXT:    v_lshrrev_b64 v[0:1], v8, v[0:1]
; CHECK-NEXT:    v_lshrrev_b64 v[2:3], v10, v[2:3]
; CHECK-NEXT:    v_lshrrev_b64 v[4:5], v11, v[4:5]
; CHECK-NEXT:    v_lshrrev_b64 v[6:7], v9, v[6:7]
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %or = or <4 x i64> %shift_amt, splat (i64 16)
  %srl = lshr <4 x i64> %arg0, %or
  ret <4 x i64> %srl
}

; test SGPR

define i64 @srl_or16_sgpr(i64 inreg %arg0, i64 inreg %shift_amt) {
; CHECK-LABEL: srl_or16_sgpr:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_or_b32 s4, s18, 16
; CHECK-NEXT:    s_lshr_b64 s[4:5], s[16:17], s4
; CHECK-NEXT:    v_mov_b32_e32 v0, s4
; CHECK-NEXT:    v_mov_b32_e32 v1, s5
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %or = or i64 %shift_amt, 16
  %srl = lshr i64 %arg0, %or
  ret i64 %srl
}

define amdgpu_ps i64 @srl_or16_sgpr_return(i64 inreg %arg0, i64 inreg %shift_amt) {
; CHECK-LABEL: srl_or16_sgpr_return:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_or_b32 s2, s2, 16
; CHECK-NEXT:    s_lshr_b64 s[0:1], s[0:1], s2
; CHECK-NEXT:    ; return to shader part epilog
  %or = or i64 %shift_amt, 16
  %srl = lshr i64 %arg0, %or
  ret i64 %srl
}

define <2 x i64> @srl_v2_or16_sgpr(<2 x i64> inreg %arg0, <2 x i64> inreg %shift_amt) {
; CHECK-LABEL: srl_v2_or16_sgpr:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_or_b32 s6, s22, 16
; CHECK-NEXT:    s_or_b32 s4, s20, 16
; CHECK-NEXT:    s_lshr_b64 s[4:5], s[16:17], s4
; CHECK-NEXT:    s_lshr_b64 s[6:7], s[18:19], s6
; CHECK-NEXT:    v_mov_b32_e32 v0, s4
; CHECK-NEXT:    v_mov_b32_e32 v1, s5
; CHECK-NEXT:    v_mov_b32_e32 v2, s6
; CHECK-NEXT:    v_mov_b32_e32 v3, s7
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %or = or <2 x i64> %shift_amt, splat (i64 16)
  %srl = lshr <2 x i64> %arg0, %or
  ret <2 x i64> %srl
}

define <3 x i64> @srl_v3_or16_sgpr(<3 x i64> inreg %arg0, <3 x i64> inreg %shift_amt) {
; CHECK-LABEL: srl_v3_or16_sgpr:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_or_b32 s8, s26, 16
; CHECK-NEXT:    s_or_b32 s6, s24, 16
; CHECK-NEXT:    s_or_b32 s4, s22, 16
; CHECK-NEXT:    s_lshr_b64 s[4:5], s[16:17], s4
; CHECK-NEXT:    s_lshr_b64 s[6:7], s[18:19], s6
; CHECK-NEXT:    s_lshr_b64 s[8:9], s[20:21], s8
; CHECK-NEXT:    v_mov_b32_e32 v0, s4
; CHECK-NEXT:    v_mov_b32_e32 v1, s5
; CHECK-NEXT:    v_mov_b32_e32 v2, s6
; CHECK-NEXT:    v_mov_b32_e32 v3, s7
; CHECK-NEXT:    v_mov_b32_e32 v4, s8
; CHECK-NEXT:    v_mov_b32_e32 v5, s9
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %or = or <3 x i64> %shift_amt, splat (i64 16)
  %srl = lshr <3 x i64> %arg0, %or
  ret <3 x i64> %srl
}

define <4 x i64> @srl_v4_or16_sgpr(<4 x i64> inreg %arg0, <4 x i64> inreg %shift_amt) {
; CHECK-LABEL: srl_v4_or16_sgpr:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_or_b32_e32 v0, 16, v0
; CHECK-NEXT:    s_or_b32 s8, s28, 16
; CHECK-NEXT:    s_or_b32 s6, s26, 16
; CHECK-NEXT:    s_or_b32 s4, s24, 16
; CHECK-NEXT:    s_lshr_b64 s[4:5], s[16:17], s4
; CHECK-NEXT:    s_lshr_b64 s[6:7], s[18:19], s6
; CHECK-NEXT:    s_lshr_b64 s[8:9], s[20:21], s8
; CHECK-NEXT:    v_lshrrev_b64 v[6:7], v0, s[22:23]
; CHECK-NEXT:    v_mov_b32_e32 v0, s4
; CHECK-NEXT:    v_mov_b32_e32 v1, s5
; CHECK-NEXT:    v_mov_b32_e32 v2, s6
; CHECK-NEXT:    v_mov_b32_e32 v3, s7
; CHECK-NEXT:    v_mov_b32_e32 v4, s8
; CHECK-NEXT:    v_mov_b32_e32 v5, s9
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %or = or <4 x i64> %shift_amt, splat (i64 16)
  %srl = lshr <4 x i64> %arg0, %or
  ret <4 x i64> %srl
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Test range with an "or X, 32"
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; These cases are reduced because computeKnownBits() can calculate a minimum of 32
; based on the OR with 32.

define i64 @srl_or32(i64 %arg0, i64 %shift_amt) {
; CHECK-LABEL: srl_or32:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_lshrrev_b32_e32 v0, v2, v1
; CHECK-NEXT:    v_mov_b32_e32 v1, 0
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %or = or i64 %shift_amt, 32
  %srl = lshr i64 %arg0, %or
  ret i64 %srl
}

define <2 x i64> @srl_v2_or32(<2 x i64> %arg0, <2 x i64> %shift_amt) {
; CHECK-LABEL: srl_v2_or32:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_lshrrev_b32_e32 v0, v4, v1
; CHECK-NEXT:    v_lshrrev_b32_e32 v2, v6, v3
; CHECK-NEXT:    v_mov_b32_e32 v1, 0
; CHECK-NEXT:    v_mov_b32_e32 v3, 0
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %or = or <2 x i64> %shift_amt, splat (i64 32)
  %srl = lshr <2 x i64> %arg0, %or
  ret <2 x i64> %srl
}

define <3 x i64> @srl_v3_or32(<3 x i64> %arg0, <3 x i64> %shift_amt) {
; CHECK-LABEL: srl_v3_or32:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_lshrrev_b32_e32 v0, v6, v1
; CHECK-NEXT:    v_lshrrev_b32_e32 v2, v8, v3
; CHECK-NEXT:    v_lshrrev_b32_e32 v4, v10, v5
; CHECK-NEXT:    v_mov_b32_e32 v1, 0
; CHECK-NEXT:    v_mov_b32_e32 v3, 0
; CHECK-NEXT:    v_mov_b32_e32 v5, 0
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %or = or <3 x i64> %shift_amt, splat (i64 32)
  %srl = lshr <3 x i64> %arg0, %or
  ret <3 x i64> %srl
}

define <4 x i64> @srl_v4_or32(<4 x i64> %arg0, <4 x i64> %shift_amt) {
; CHECK-LABEL: srl_v4_or32:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_lshrrev_b32_e32 v0, v8, v1
; CHECK-NEXT:    v_lshrrev_b32_e32 v2, v10, v3
; CHECK-NEXT:    v_lshrrev_b32_e32 v4, v12, v5
; CHECK-NEXT:    v_lshrrev_b32_e32 v6, v14, v7
; CHECK-NEXT:    v_mov_b32_e32 v1, 0
; CHECK-NEXT:    v_mov_b32_e32 v3, 0
; CHECK-NEXT:    v_mov_b32_e32 v5, 0
; CHECK-NEXT:    v_mov_b32_e32 v7, 0
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %or = or <4 x i64> %shift_amt, splat (i64 32)
  %srl = lshr <4 x i64> %arg0, %or
  ret <4 x i64> %srl
}

; test SGPR

define i64 @srl_or32_sgpr(i64 inreg %arg0, i64 inreg %shift_amt) {
; CHECK-LABEL: srl_or32_sgpr:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_lshr_b32 s4, s17, s18
; CHECK-NEXT:    v_mov_b32_e32 v0, s4
; CHECK-NEXT:    v_mov_b32_e32 v1, 0
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %or = or i64 %shift_amt, 32
  %srl = lshr i64 %arg0, %or
  ret i64 %srl
}

define amdgpu_ps i64 @srl_or32_sgpr_return(i64 inreg %arg0, i64 inreg %shift_amt) {
; CHECK-LABEL: srl_or32_sgpr_return:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_lshr_b32 s0, s1, s2
; CHECK-NEXT:    s_mov_b32 s1, 0
; CHECK-NEXT:    ; return to shader part epilog
  %or = or i64 %shift_amt, 32
  %srl = lshr i64 %arg0, %or
  ret i64 %srl
}

define <2 x i64> @srl_v2_or32_sgpr(<2 x i64> inreg %arg0, <2 x i64> inreg %shift_amt) {
; CHECK-LABEL: srl_v2_or32_sgpr:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_lshr_b32 s4, s17, s20
; CHECK-NEXT:    s_lshr_b32 s5, s19, s22
; CHECK-NEXT:    v_mov_b32_e32 v0, s4
; CHECK-NEXT:    v_mov_b32_e32 v1, 0
; CHECK-NEXT:    v_mov_b32_e32 v2, s5
; CHECK-NEXT:    v_mov_b32_e32 v3, 0
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %or = or <2 x i64> %shift_amt, splat (i64 32)
  %srl = lshr <2 x i64> %arg0, %or
  ret <2 x i64> %srl
}

define <3 x i64> @srl_v3_or32_sgpr(<3 x i64> inreg %arg0, <3 x i64> inreg %shift_amt) {
; CHECK-LABEL: srl_v3_or32_sgpr:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_lshr_b32 s4, s17, s22
; CHECK-NEXT:    s_lshr_b32 s5, s19, s24
; CHECK-NEXT:    s_lshr_b32 s6, s21, s26
; CHECK-NEXT:    v_mov_b32_e32 v0, s4
; CHECK-NEXT:    v_mov_b32_e32 v1, 0
; CHECK-NEXT:    v_mov_b32_e32 v2, s5
; CHECK-NEXT:    v_mov_b32_e32 v3, 0
; CHECK-NEXT:    v_mov_b32_e32 v4, s6
; CHECK-NEXT:    v_mov_b32_e32 v5, 0
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %or = or <3 x i64> %shift_amt, splat (i64 32)
  %srl = lshr <3 x i64> %arg0, %or
  ret <3 x i64> %srl
}

define <4 x i64> @srl_v4_or32_sgpr(<4 x i64> inreg %arg0, <4 x i64> inreg %shift_amt) {
; CHECK-LABEL: srl_v4_or32_sgpr:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_lshr_b32 s4, s17, s24
; CHECK-NEXT:    s_lshr_b32 s5, s19, s26
; CHECK-NEXT:    s_lshr_b32 s6, s21, s28
; CHECK-NEXT:    v_lshrrev_b32_e64 v6, v0, s23
; CHECK-NEXT:    v_mov_b32_e32 v0, s4
; CHECK-NEXT:    v_mov_b32_e32 v1, 0
; CHECK-NEXT:    v_mov_b32_e32 v2, s5
; CHECK-NEXT:    v_mov_b32_e32 v3, 0
; CHECK-NEXT:    v_mov_b32_e32 v4, s6
; CHECK-NEXT:    v_mov_b32_e32 v5, 0
; CHECK-NEXT:    v_mov_b32_e32 v7, 0
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %or = or <4 x i64> %shift_amt, splat (i64 32)
  %srl = lshr <4 x i64> %arg0, %or
  ret <4 x i64> %srl
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Test range from max/min
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; FIXME: This case should be reduced too, but computeKnownBits() cannot
;        determine the range.  Match current results for now.

define i64 @srl_maxmin(i64 %arg0, i64 noundef %arg1) {
; CHECK-LABEL: srl_maxmin:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_cmp_lt_u64_e32 vcc, 32, v[2:3]
; CHECK-NEXT:    v_cndmask_b32_e32 v3, 0, v3, vcc
; CHECK-NEXT:    v_cndmask_b32_e32 v2, 32, v2, vcc
; CHECK-NEXT:    v_cmp_gt_u64_e32 vcc, 63, v[2:3]
; CHECK-NEXT:    v_cndmask_b32_e32 v2, 63, v2, vcc
; CHECK-NEXT:    v_lshrrev_b64 v[0:1], v2, v[0:1]
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %max = call i64 @llvm.umax.i64(i64 %arg1, i64 32)
  %min = call i64 @llvm.umin.i64(i64 %max,  i64 63)
  %srl = lshr i64 %arg0, %min
  ret i64 %srl
}

define <2 x i64> @srl_v2_maxmin(<2 x i64> %arg0, <2 x i64> noundef %arg1) {
; CHECK-LABEL: srl_v2_maxmin:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_cmp_lt_u64_e32 vcc, 32, v[4:5]
; CHECK-NEXT:    v_cndmask_b32_e32 v5, 0, v5, vcc
; CHECK-NEXT:    v_cndmask_b32_e32 v4, 32, v4, vcc
; CHECK-NEXT:    v_cmp_lt_u64_e32 vcc, 32, v[6:7]
; CHECK-NEXT:    v_cndmask_b32_e32 v7, 0, v7, vcc
; CHECK-NEXT:    v_cndmask_b32_e32 v6, 32, v6, vcc
; CHECK-NEXT:    v_cmp_gt_u64_e32 vcc, 63, v[6:7]
; CHECK-NEXT:    v_cndmask_b32_e32 v6, 63, v6, vcc
; CHECK-NEXT:    v_cmp_gt_u64_e32 vcc, 63, v[4:5]
; CHECK-NEXT:    v_lshrrev_b64 v[2:3], v6, v[2:3]
; CHECK-NEXT:    v_cndmask_b32_e32 v4, 63, v4, vcc
; CHECK-NEXT:    v_lshrrev_b64 v[0:1], v4, v[0:1]
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %max = call <2 x i64> @llvm.umax.i64(<2 x i64> %arg1, <2 x i64> splat (i64 32))
  %min = call <2 x i64> @llvm.umin.i64(<2 x i64> %max,  <2 x i64> splat (i64 63))
  %srl = lshr <2 x i64> %arg0, %min
  ret <2 x i64> %srl
}

define <3 x i64> @srl_v3_maxmin(<3 x i64> %arg0, <3 x i64> noundef %arg1) {
; CHECK-LABEL: srl_v3_maxmin:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_cmp_lt_u64_e32 vcc, 32, v[6:7]
; CHECK-NEXT:    v_cndmask_b32_e32 v7, 0, v7, vcc
; CHECK-NEXT:    v_cndmask_b32_e32 v6, 32, v6, vcc
; CHECK-NEXT:    v_cmp_lt_u64_e32 vcc, 32, v[8:9]
; CHECK-NEXT:    v_cndmask_b32_e32 v9, 0, v9, vcc
; CHECK-NEXT:    v_cndmask_b32_e32 v8, 32, v8, vcc
; CHECK-NEXT:    v_cmp_lt_u64_e32 vcc, 32, v[10:11]
; CHECK-NEXT:    v_cndmask_b32_e32 v11, 0, v11, vcc
; CHECK-NEXT:    v_cndmask_b32_e32 v10, 32, v10, vcc
; CHECK-NEXT:    v_cmp_gt_u64_e32 vcc, 63, v[10:11]
; CHECK-NEXT:    v_cndmask_b32_e32 v10, 63, v10, vcc
; CHECK-NEXT:    v_cmp_gt_u64_e32 vcc, 63, v[8:9]
; CHECK-NEXT:    v_lshrrev_b64 v[4:5], v10, v[4:5]
; CHECK-NEXT:    v_cndmask_b32_e32 v8, 63, v8, vcc
; CHECK-NEXT:    v_cmp_gt_u64_e32 vcc, 63, v[6:7]
; CHECK-NEXT:    v_lshrrev_b64 v[2:3], v8, v[2:3]
; CHECK-NEXT:    v_cndmask_b32_e32 v6, 63, v6, vcc
; CHECK-NEXT:    v_lshrrev_b64 v[0:1], v6, v[0:1]
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %max = call <3 x i64> @llvm.umax.i64(<3 x i64> %arg1, <3 x i64> splat (i64 32))
  %min = call <3 x i64> @llvm.umin.i64(<3 x i64> %max,  <3 x i64> splat (i64 63))
  %srl = lshr <3 x i64> %arg0, %min
  ret <3 x i64> %srl
}

define <4 x i64> @srl_v4_maxmin(<4 x i64> %arg0, <4 x i64> noundef %arg1) {
; CHECK-LABEL: srl_v4_maxmin:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_cmp_lt_u64_e32 vcc, 32, v[8:9]
; CHECK-NEXT:    v_cndmask_b32_e32 v9, 0, v9, vcc
; CHECK-NEXT:    v_cndmask_b32_e32 v8, 32, v8, vcc
; CHECK-NEXT:    v_cmp_lt_u64_e32 vcc, 32, v[10:11]
; CHECK-NEXT:    v_cndmask_b32_e32 v11, 0, v11, vcc
; CHECK-NEXT:    v_cndmask_b32_e32 v10, 32, v10, vcc
; CHECK-NEXT:    v_cmp_lt_u64_e32 vcc, 32, v[12:13]
; CHECK-NEXT:    v_cndmask_b32_e32 v13, 0, v13, vcc
; CHECK-NEXT:    v_cndmask_b32_e32 v12, 32, v12, vcc
; CHECK-NEXT:    v_cmp_lt_u64_e32 vcc, 32, v[14:15]
; CHECK-NEXT:    v_cndmask_b32_e32 v15, 0, v15, vcc
; CHECK-NEXT:    v_cndmask_b32_e32 v14, 32, v14, vcc
; CHECK-NEXT:    v_cmp_gt_u64_e32 vcc, 63, v[14:15]
; CHECK-NEXT:    v_cndmask_b32_e32 v14, 63, v14, vcc
; CHECK-NEXT:    v_cmp_gt_u64_e32 vcc, 63, v[12:13]
; CHECK-NEXT:    v_lshrrev_b64 v[6:7], v14, v[6:7]
; CHECK-NEXT:    v_cndmask_b32_e32 v12, 63, v12, vcc
; CHECK-NEXT:    v_cmp_gt_u64_e32 vcc, 63, v[10:11]
; CHECK-NEXT:    v_lshrrev_b64 v[4:5], v12, v[4:5]
; CHECK-NEXT:    v_cndmask_b32_e32 v10, 63, v10, vcc
; CHECK-NEXT:    v_cmp_gt_u64_e32 vcc, 63, v[8:9]
; CHECK-NEXT:    v_lshrrev_b64 v[2:3], v10, v[2:3]
; CHECK-NEXT:    v_cndmask_b32_e32 v8, 63, v8, vcc
; CHECK-NEXT:    v_lshrrev_b64 v[0:1], v8, v[0:1]
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %max = call <4 x i64> @llvm.umax.i64(<4 x i64> %arg1, <4 x i64> splat (i64 32))
  %min = call <4 x i64> @llvm.umin.i64(<4 x i64> %max,  <4 x i64> splat (i64 63))
  %srl = lshr <4 x i64> %arg0, %min
  ret <4 x i64> %srl
}
