; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 3
; RUN: opt -S < %s --passes=slp-vectorizer | FileCheck %s

define i1 @test(i32 %x) {
; CHECK-LABEL: define i1 @test(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[CMP:%.*]] = icmp sgt i32 [[X]], 1
; CHECK-NEXT:    [[OP_RDX:%.*]] = select i1 [[CMP]], i1 true, i1 poison
; CHECK-NEXT:    ret i1 [[OP_RDX]]
;
  %cmp = icmp sgt i32 %x, 1
  %sel1 = select i1 %cmp, i1 true, i1 poison
  %sel2 = select i1 %sel1, i1 true, i1 poison
  %sel3 = select i1 %sel2, i1 true, i1 poison
  %sel4 = select i1 %cmp, i1 true, i1 poison
  %ret = or i1 %sel3, %sel4
  ret i1 %ret
}

define i1 @test1(i32 %x, i32 %a, i32 %b, i32 %c, i32 %d) {
; CHECK-LABEL: define i1 @test1(
; CHECK-SAME: i32 [[X:%.*]], i32 [[A:%.*]], i32 [[B:%.*]], i32 [[C:%.*]], i32 [[D:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = insertelement <4 x i32> poison, i32 [[X]], i32 0
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <4 x i32> [[TMP1]], i32 [[A]], i32 1
; CHECK-NEXT:    [[TMP3:%.*]] = insertelement <4 x i32> [[TMP2]], i32 [[B]], i32 2
; CHECK-NEXT:    [[TMP4:%.*]] = insertelement <4 x i32> [[TMP3]], i32 [[C]], i32 3
; CHECK-NEXT:    [[TMP5:%.*]] = icmp sgt <4 x i32> [[TMP4]], splat (i32 1)
; CHECK-NEXT:    [[CMP3:%.*]] = icmp sgt i32 [[D]], 1
; CHECK-NEXT:    [[TMP6:%.*]] = freeze <4 x i1> [[TMP5]]
; CHECK-NEXT:    [[TMP7:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[TMP6]])
; CHECK-NEXT:    [[OP_RDX:%.*]] = select i1 [[TMP7]], i1 true, i1 [[CMP3]]
; CHECK-NEXT:    ret i1 [[OP_RDX]]
;
  %cmp = icmp sgt i32 %x, 1
  %cmp1 = icmp sgt i32 %a, 1
  %cmp2 = icmp sgt i32 %b, 1
  %cmp3 = icmp sgt i32 %c, 1
  %cmp4 = icmp sgt i32 %d, 1
  %sel1 = select i1 %cmp, i1 true, i1 %cmp1
  %sel2 = select i1 %sel1, i1 true, i1 %cmp2
  %sel3 = select i1 %sel2, i1 true, i1 %cmp3
  %sel4 = select i1 %cmp, i1 true, i1 %cmp4
  %ret = or i1 %sel3, %sel4
  ret i1 %ret
}
