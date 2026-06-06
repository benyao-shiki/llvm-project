# CudaKernelAnalysis Pass

## Overview

`CudaKernelAnalysis` 是面向 CUDA kernel 的 LLVM 分析 pass，用于给参数路径赋权，供后续 `CudaKernelConst` 与 `CudaKernelNoalias` 选择候选集。

当前实现统一使用“参数路径”表示法：

- 顶层参数：`[ArgIndex, 0]`
- 结构体成员：`[ArgIndex, ByteOffset]`
- CUDA 隐式维度参数：`[SPECIAL_INDEX, 0]`

其中 `ByteOffset` 是相对顶层参数起始地址的字节偏移。

## Kernel 识别

只分析 CUDA kernel 函数：

- `CallingConv::PTX_Kernel`，或
- 函数属性包含 `ptx.kernel`

非 kernel 函数直接跳过。

## 分析对象

统一覆盖以下对象：

1. 顶层标量参数（整型/浮点）
2. 顶层指针参数
3. 结构体按值参数（含嵌套成员）
4. 指针链访问到的结构体成员（按 GEP 常量偏移归一）
5. CUDA 维度 intrinsic：
   - `gridDim.{x,y,z}`
   - `blockDim.{x,y,z}`

## 权重规则（当前实现）

### Scalar 权重（`analyzeScalarValue`）

初始值：`1`

叠加规则：

- 直接参与 `br/switch`：`+10`
- 先比较、比较结果再参与分支：`+5`
- 在循环体内使用：`+3`
- 作为 `store` 的 value operand：`+4`
- 参与二元算术/位运算：`+2`

### Pointer 权重（`analyzePointerValue`）

初始值：`1`（当前分支用于放宽候选覆盖）

叠加规则：

- 在循环体内使用：`+5`
- 参与 `load/store`：`+3`
- 参与 `GEP` 指针运算：`+2`
- 对所在循环满足 loop-invariant：`+2`

## 结构体与偏移路径

实现核心是按 `DataLayout` + `accumulateConstantOffset` 计算总字节偏移：

- 对 byval / 指针基访问路径：沿 `GEP -> (bitcast) -> load` 递归追踪
- 对直接按值 struct：结合 `ExtractValueInst` 的逻辑索引和 `StructLayout` 转换到字节偏移

因此，不同 IR 形态会被统一映射到同一 `[ArgIndex, ByteOffset]` 路径。

## 维度参数建模

Pass 会扫描以下 intrinsic 调用：

- `llvm.nvvm.read.ptx.sreg.nctaid.{x,y,z}`
- `llvm.nvvm.read.ptx.sreg.ntid.{x,y,z}`

若该 intrinsic 计算得到的权重 `> 0`，会将其累加到 `ScalarWeights`，路径首元素为保留的 `SPECIAL_INDEX_*`。

## 与下游 Pass 的关系

- `CudaKernelConst` 读取 `ScalarWeights`，再叠加 profile 统计做候选子集搜索。
- `CudaKernelNoalias` 读取 `PointerWeights`，再叠加 profile alias 比例做候选子集搜索。
- 候选是否进入搜索由下游阈值控制：
  - `-mllvm -const-min-weight`
  - `-mllvm -noalias-min-weight`
- 同时开启 const/noalias 时，下游不会在 analysis pass 中重新计算联合权重；联合版本 `<orig>_const_noalias` 复用两个 pass 各自已经选择出的 scalar/pointer 路径。

## 联合优化中的角色

联合优化的设备端生成顺序为：

1. `CudaKernelConstPass` 根据 `ScalarWeights` 选择常量路径并生成 `<orig>_const`
2. `CudaKernelNoaliasPass` 根据 `PointerWeights` 选择 noalias 路径并生成 `<orig>_noalias`
3. 若 `<orig>_const` 已存在，则 noalias pass 继续生成 `<orig>_const_noalias`

因此，`CudaKernelAnalysis` 仍只负责提供两类独立权重：

- `ScalarWeights` 面向常量传播
- `PointerWeights` 面向 noalias

组合版本不是新的第三类分析结果，而是在代码生成和动态分流阶段把两个独立优化结果组合起来。

## 调试

可通过：

```bash
./bin/opt -passes='require<cuda-kernel-analysis>' \
  -debug-only=cuda-kernel-analysis -disable-output input.ll
```

输出会显示：

- Scalar/Pointer 路径与权重
- 路径格式（如 `indexPath=1.4`）
- 维度 intrinsic 的识别与计权情况
