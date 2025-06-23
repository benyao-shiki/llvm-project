# CUDA Kernel Noalias Optimization

## 概述

这是一个为LLVM/Clang添加的CUDA kernel优化功能，可以自动为具有多个指针参数的CUDA kernel函数创建带有noalias属性的克隆版本，从而允许编译器进行更积极的优化。

## 功能特性

- 自动识别CUDA kernel函数
- 为具有2个或更多指针参数的kernel创建noalias克隆版本
- 通过编译器选项控制功能启用/禁用
- 集成到标准CUDA编译流程中

## 实现详情

### 1. LLVM Pass实现
- **位置**: `llvm/lib/Transforms/CudaKernelNoalias/`
- **核心文件**: `CudaKernelNoalias.cpp`, `CudaKernelNoalias.h`
- **功能**: 
  - 检测`CallingConv::PTX_Kernel`调用约定的函数
  - 分析指针参数数量
  - 为符合条件的kernel创建带有noalias属性的克隆函数
  - 复制nvvm.annotations元数据

### 2. 编译器选项
- **选项定义**: `clang/include/clang/Driver/Options.td`
- **选项**: `-fcuda-kernel-noalias` / `-fno-cuda-kernel-noalias`
- **默认状态**: 禁用

### 3. 集成点
- **后端集成**: `clang/lib/CodeGen/BackendUtil.cpp`
- **Driver传递**: `clang/lib/Driver/ToolChains/Clang.cpp`, `clang/lib/Driver/ToolChains/Cuda.cpp`
- **Pass注册**: `llvm/lib/Passes/PassRegistry.def`, `llvm/lib/Passes/PassBuilder.cpp`

## 使用方法

### 编译选项
```bash
# 启用CUDA kernel noalias优化
clang++ -fcuda-kernel-noalias --cuda-gpu-arch=sm_XX source.cu -o output

# 禁用优化（默认）
clang++ -fno-cuda-kernel-noalias --cuda-gpu-arch=sm_XX source.cu -o output
```

### 示例代码
```cuda
// 这个kernel有3个指针参数，会创建noalias克隆版本
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 这个kernel只有1个指针参数，不会创建克隆版本
__global__ void fill_array(float* data, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}
```

## 验证方法

### 1. 检查生成的函数
```bash
# 编译程序
clang++ -fcuda-kernel-noalias --cuda-gpu-arch=sm_52 test.cu -o test -lcudart

# 检查是否生成了noalias版本
strings test | grep "_noalias$"
```

预期输出：
```
_Z10vector_addPfS_S_i_noalias
```

### 2. 查看LLVM IR
```bash
# 生成设备端IR
clang++ -fcuda-kernel-noalias --cuda-gpu-arch=sm_52 --cuda-device-only -S -emit-llvm test.cu -o test.ll

# 查看noalias属性
grep -E "define.*noalias|ptr noalias" test.ll
```

预期输出：
```
define dso_local ptx_kernel void @_Z10vector_addPfS_S_i_noalias(ptr noalias noundef readonly %a, ptr noalias noundef readonly %b, ptr noalias noundef writeonly %c, i32 noundef %n)
```

### 3. 对比测试
```bash
# 启用优化版本
clang++ -fcuda-kernel-noalias --cuda-gpu-arch=sm_52 test.cu -o test_enabled -lcudart
strings test_enabled | grep -c "_noalias$"

# 禁用优化版本  
clang++ -fno-cuda-kernel-noalias --cuda-gpu-arch=sm_52 test.cu -o test_disabled -lcudart
strings test_disabled | grep -c "_noalias$"
```

启用版本应该输出非零数量，禁用版本应该输出0。

## 技术细节

### Pass执行时机
- 在设备端编译阶段执行
- 通过`PipelineStartEPCallback`注册到优化流水线开始处

### 克隆策略
- 为每个有2+指针参数的kernel创建一个noalias克隆版本
- 对所有指针参数添加noalias属性
- 保持原始kernel函数不变

### 命名规则
- 原函数名 + `_noalias` 后缀
- 例如：`vector_add` → `vector_add_noalias`

## 构建要求

需要重新构建LLVM/Clang以包含此功能：

```bash
cd llvm-project/build
ninja -j$(nproc)
```

## 文件修改列表

### 新增文件
- `llvm/include/llvm/Transforms/CudaKernelNoalias/CudaKernelNoalias.h`
- `llvm/lib/Transforms/CudaKernelNoalias/CudaKernelNoalias.cpp`
- `llvm/lib/Transforms/CudaKernelNoalias/CMakeLists.txt`

### 修改文件
- `clang/include/clang/Driver/Options.td`
- `clang/include/clang/Basic/CodeGenOptions.def`
- `clang/lib/CodeGen/BackendUtil.cpp`
- `clang/lib/Driver/ToolChains/Clang.cpp`
- `clang/lib/Driver/ToolChains/Cuda.cpp`
- `llvm/lib/Transforms/CMakeLists.txt`
- `llvm/lib/Passes/PassRegistry.def`
- `llvm/lib/Passes/PassBuilder.cpp`
- `llvm/lib/Passes/CMakeLists.txt`
- `llvm/tools/opt/CMakeLists.txt`

## 性能影响

- **编译时间**: 轻微增加，主要来自函数克隆
- **二进制大小**: 每个符合条件的kernel增加一个克隆版本
- **运行时性能**: 通过noalias属性可能获得更好的优化效果

## 限制

- 仅适用于有2个或更多指针参数的kernel函数
- 需要CUDA编译环境
- 克隆函数的实际使用需要运行时或链接时选择机制（未实现）

## 未来改进

- 添加运行时自动选择机制
- 根据指针别名分析结果智能决定是否克隆
- 支持更细粒度的noalias控制
- 添加性能分析和度量工具 