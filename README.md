# CudaKernelAnalysis Pass

## 概述

CudaKernelAnalysis 是一个LLVM分析pass，专门用于分析CUDA内核函数的参数特性，为后续的优化pass提供决策依据。

## 功能特性

### 1. CUDA内核识别
- 通过 `ptx.kernel` 属性识别CUDA内核函数
- 通过 `PTX_Kernel` 调用约定识别CUDA内核函数
- 自动跳过非CUDA内核函数的分析

### 2. 标量参数权重分析
分析标量参数（整数、浮点数）在控制流中的重要性：
- **分支指令使用**：直接用于 `br` 或 `switch` 指令的参数权重 +10
- **比较指令使用**：用于比较指令且结果用于分支的参数权重 +5
- 权重越高表示参数在控制流中越重要

### 3. 指针参数noalias收益分析
分析指针参数添加 `noalias` 属性的潜在收益：
- 检查指针是否在循环中被使用
- 如果在循环中使用，则认为添加 `noalias` 可能有益

## 使用方法

### 1. 基本使用

```bash
# 在pass pipeline中使用
./bin/opt -passes='require<cuda-kernel-analysis>' -disable-output input.ll

# 获取调试输出
./bin/opt -passes='require<cuda-kernel-analysis>' -debug-only=cuda-kernel-analysis -disable-output input.ll
```

### 2. 与其他pass配合使用

```bash
# 在transform pass之前使用
./bin/opt -passes='require<cuda-kernel-analysis>,your-transform-pass' input.ll

# 在多个pass中使用
./bin/opt -passes='require<cuda-kernel-analysis>,cuda-kernel-noalias,cuda-kernel-const' input.ll
```

### 3. 调试选项

```bash
# 启用所有调试输出
./bin/opt -debug -passes='require<cuda-kernel-analysis>' input.ll

# 只启用cuda-kernel-analysis的调试输出
./bin/opt -debug-only=cuda-kernel-analysis -passes='require<cuda-kernel-analysis>' input.ll
```

## 输出示例

### 调试输出示例

```
CudaKernelAnalysis: Analyzing CUDA kernel function 'my_cuda_kernel'
  Pointer argument 'data': noalias benefit = true
  Scalar argument 'size': weight = 10
  Scalar argument 'threshold': weight = 5
CudaKernelAnalysis: Analysis completed for function 'my_cuda_kernel'
```

### 非CUDA内核函数

```
CudaKernelAnalysis: Function 'normal_function' is not a CUDA kernel, skipping analysis
```

## 分析结果

### CudaKernelAnalysisResult 结构

```cpp
class CudaKernelAnalysisResult {
  // 标量参数权重映射
  DenseMap<const Argument *, int> ScalarWeights;
  
  // 指针参数noalias收益映射
  DenseMap<const Argument *, bool> PointerBenefits;
  
  // 调试输出标志
  bool DebugOutput;
};
```

### 在其他pass中使用

```cpp
// 获取分析结果
auto &Result = FAM.getResult<CudaKernelAnalysis>(F);

// 使用标量权重
for (auto [Arg, Weight] : Result.ScalarWeights) {
  if (Weight > 5) {
    // 高权重参数的处理逻辑
  }
}

// 使用指针收益信息
for (auto [Arg, Benefit] : Result.PointerBenefits) {
  if (Benefit) {
    // 添加noalias属性的逻辑
  }
}
```

## 示例CUDA内核

```llvm
define void @my_cuda_kernel(i32* %data, i32 %size, i32 %threshold) #0 {
entry:
  %0 = icmp sgt i32 %size, 0          ; size参数用于比较
  br i1 %0, label %loop, label %exit  ; 比较结果用于分支

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.latch ]
  %val = load i32, i32* %data, align 4  ; data指针在循环中使用
  %cmp = icmp slt i32 %val, %threshold  ; threshold参数用于比较
  br i1 %cmp, label %if.then, label %if.else  ; 比较结果用于分支
  ; ... 更多代码
}

attributes #0 = { "ptx.kernel"="true" }
```

分析结果：
- `data` (指针): noalias benefit = true (在循环中使用)
- `size` (标量): weight = 10 (直接用于分支)
- `threshold` (标量): weight = 5 (用于比较且结果用于分支)

## 编译和安装

### 编译LLVM

```bash
# 在LLVM源码目录中
mkdir build && cd build
cmake -G Ninja -DLLVM_ENABLE_PROJECTS="clang;lld" -DCMAKE_BUILD_TYPE=Release ../llvm
ninja
```

### 验证安装

```bash
# 检查pass是否可用
./bin/opt --print-passes | grep cuda-kernel-analysis

# 测试pass功能
./bin/opt -passes='require<cuda-kernel-analysis>' -debug-only=cuda-kernel-analysis -disable-output test.ll
```

## 注意事项

1. **性能影响**：分析pass只在需要时运行，不会影响正常编译性能
2. **调试输出**：使用 `-debug-only=cuda-kernel-analysis` 可以获取详细的调试信息
3. **依赖关系**：该pass依赖 `LoopAnalysis` 来分析指针在循环中的使用情况
4. **兼容性**：支持所有LLVM支持的CUDA内核识别方式

## 扩展性

该pass设计为可扩展的：
- 可以添加更多的权重计算规则
- 可以扩展noalias收益分析算法
- 可以添加其他类型的参数分析

## 贡献

欢迎提交issue和pull request来改进这个pass的功能和性能。
