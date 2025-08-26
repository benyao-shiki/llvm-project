# CudaKernelAnalysis Pass

## Overview

CudaKernelAnalysis 是一个针对 CUDA kernel 的 LLVM 分析 pass，用于分析内核参数（包含按值结构体成员）的使用形态与重要性，为后续优化（如常量传播、noalias）提供权重依据与成员路径信息。

该 pass 的核心是将参数成员唯一地标识为其**顶层参数索引**和从该参数开始的**字节偏移量**。

## Features

### 1. CUDA Kernel Identification
- 通过调用约定 `PTX_Kernel` 或函数属性 `"ptx.kernel"` 识别 CUDA kernel。
- 自动跳过非 CUDA kernel 函数。

### 2. Unified Parameter Analysis
统一分析三类：
- 标量参数（整数/浮点）
- 指针参数（用于 noalias）
- 结构体成员（支持按值传递的结构体与嵌套成员）

成员路径以 `[ArgIndex, ByteOffset]` 的形式表示。例如：`[1, 4]` 表示第 1 个参数（从0计数）的偏移量为 4 字节的成员。

### 3. Weight Rules（权重规则）
- 标量权重（控制流重要性）：
  - 直接用于 `br`/`switch`：+10
  - 用于比较且比较结果用于分支：+5
  - 循环内使用：+3
  - 作为 `store` 的被存储值：+4
  - 参与二元（算术/位）运算：+2
- 指针权重（noalias 潜力）：
  - 循环内使用：+5
  - load/store：+3
  - GEP 指针运算：+2
  - 循环不变（Loop-invariant）：+2

更高的权重代表更值得进行专门化或约束。

## Implementation Details

### 1. Struct(byval) 与指向结构体的指针
- 对于任何基于指针的访问（无论是普通指针还是 `byval` 参数），分析都会追踪 `getelementptr` (GEP) 指令。
- **核心逻辑**：通过 `GEP->accumulateConstantOffset()` 计算出每个被访问成员相对于其顶层参数起始地址的**总字节偏移量**。
- 这种基于偏移量的方法确保了无论结构体在 IR 中如何布局（例如被扁平化），或者 GEP 是基于 `i8` 还是结构体类型，我们都能得到一个稳定且唯一的标识符。
- 递归处理嵌套结构体：在递归进入下一层结构体时，当前计算出的偏移量会被传递下去，以确保最终叶成员的偏移量是相对于最外层参数的绝对偏移。

### 2. 直接按值结构体参数（非指针）
- 若函数签名参数本身是 `struct`（by value），分析会遍历 `extractvalue` 指令。由于 `extractvalue` 直接使用逻辑索引，分析会结合 `DataLayout` 将这些逻辑索引转换为相应的字节偏移量，以保持与基于指针的分析统一。

### 3. 路径与显示名
- `ParameterInfo` 保存 `Indices`，其格式为 `[ArgIndex, ByteOffset]`。
- `getDisplayName()` 用于调试打印，结构体成员以 `name[offset]` 的形式展示。

### 4. Debug 输出
开启 `-debug-only=cuda-kernel-analysis` 可看到：
- 顶层指针、标量参数的权重。
- 结构体成员（含 byval）经 GEP/Load 识别并计权的明细，以偏移量（offset）而非索引路径（path）展示。
- 汇总列表：
  - Scalar arguments (including struct members)
  - Pointer arguments (including struct members)

示例（`test_const.cu`）：
```
CudaKernelAnalysis: Analyzing CUDA kernel function 'fill_runtime'
  Pointer argument 'x': weight = 9
  Struct scalar member offset=4: weight = 5
  Scalar argument 'size': weight = 8
  Scalar arguments (including struct members):
    - indexPath=1.4 name='s[4]' weight=5
    - indexPath=2 name='size' weight=8
  Pointer arguments (including struct members):
    - indexPath=0 name='x' weight=9
CudaKernelAnalysis: Analysis completed for function 'fill_runtime'
```

## Usage

### Run with opt
```bash
./bin/opt -passes='require<cuda-kernel-analysis>' -disable-output input.ll
# Debug
./bin/opt -passes='require<cuda-kernel-analysis>' -debug-only=cuda-kernel-analysis -disable-output input.ll
```

### Integrate with other passes
```bash
# Before transform passes
./bin/opt -passes='require<cuda-kernel-analysis>,cuda-kernel-const' input.ll
```

## Test Case: `test_const.cu`

```cuda
struct MyStruct { int a; int value; };
extern "C" __global__ void fill_runtime(int *x, MyStruct s, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = idx; i < size; i += gridDim.x * blockDim.x) {
    x[i] *= s.value;
  }
}
```

- 期望结果：
  - `x`（指针，参数0）获得较高指针权重。
  - `s.value`（struct 成员）：`s` 是参数 1，`value` 在 `a` (4字节)之后，所以其偏移量为 4。分析将产生路径 `[1, 4]` 并为其赋权。
  - `size`（标量，参数2）获得较高标量权重。

这些结果将作为 `CudaKernelConstPass` 候选集合的基础，后者据此在 `kernels[]` 统计中选择最优常量组合并进行克隆与常量替换。

## Notes

1. 依赖 `LoopAnalysis` 以识别循环内使用与不变性。
2. 通过 `accumulateConstantOffset` 统一处理不同形式的 GEP，无需区分 `i8` 或结构体类型，增强了稳健性。
