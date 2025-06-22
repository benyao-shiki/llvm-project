# The LLVM Compiler Infrastructure

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/llvm/llvm-project/badge)](https://securityscorecards.dev/viewer/?uri=github.com/llvm/llvm-project)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8273/badge)](https://www.bestpractices.dev/projects/8273)
[![libc++](https://github.com/llvm/llvm-project/actions/workflows/libcxx-build-and-test.yaml/badge.svg?branch=main&event=schedule)](https://github.com/llvm/llvm-project/actions/workflows/libcxx-build-and-test.yaml?query=event%3Aschedule)

Welcome to the LLVM project!

This repository contains the source code for LLVM, a toolkit for the
construction of highly optimized compilers, optimizers, and run-time
environments.

The LLVM project has multiple components. The core of the project is
itself called "LLVM". This contains all of the tools, libraries, and header
files needed to process intermediate representations and convert them into
object files. Tools include an assembler, disassembler, bitcode analyzer, and
bitcode optimizer.

C-like languages use the [Clang](https://clang.llvm.org/) frontend. This
component compiles C, C++, Objective-C, and Objective-C++ code into LLVM bitcode
-- and from there into object files, using LLVM.

Other components include:
the [libc++ C++ standard library](https://libcxx.llvm.org),
the [LLD linker](https://lld.llvm.org), and more.

## Getting the Source Code and Building LLVM

Consult the
[Getting Started with LLVM](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm)
page for information on building and running LLVM.

For information on how to contribute to the LLVM project, please take a look at
the [Contributing to LLVM](https://llvm.org/docs/Contributing.html) guide.

## Getting in touch

Join the [LLVM Discourse forums](https://discourse.llvm.org/), [Discord
chat](https://discord.gg/xS7Z362),
[LLVM Office Hours](https://llvm.org/docs/GettingInvolved.html#office-hours) or
[Regular sync-ups](https://llvm.org/docs/GettingInvolved.html#online-sync-ups).

The LLVM project has adopted a [code of conduct](https://llvm.org/docs/CodeOfConduct.html) for
participants to all modes of communication within the project.

# CUDA Kernel Arguments Profiling Tool

一个基于LLVM的CUDA内核参数自动分析工具，能够在编译时自动插入instrumentation代码，在运行时记录CUDA内核的启动参数、网格/块维度以及标量参数的名称和值。

## 🎯 功能特性

- **自动化instrumentation**：编译时自动插入profiling代码，无需修改源代码
- **智能参数识别**：自动区分标量参数和指针参数，只记录标量值
- **参数名称保留**：从LLVM IR中提取原始参数名称
- **内核名称解析**：自动解析和简化CUDA内核函数名
- **类型智能检测**：自动识别整数和浮点数类型
- **零运行时开销**：只在启用profiling时产生开销

## 📋 输出示例

```
CUDA Kernel Arguments Profile Log
===================================

Kernel Launch: vector_add
  Grid Dimensions: (4, 1, 1)
  Block Dimensions: (256, 1, 1)
  Scalar Argument: n = 1000

Kernel Launch: scalar_multiply
  Grid Dimensions: (8, 1, 1)
  Block Dimensions: (128, 1, 1)
  Scalar Argument: scalar = 2.500000
  Scalar Argument: size = 1000
```

## 🏗️ 原理

### 编译时组件

1. **LLVM Transform Pass** (`CudaArgsProfile`)
   - 在LLVM IR层面分析CUDA程序
   - 识别`cudaLaunchKernel()`调用
   - 插入profiling函数调用
   - 智能过滤标量vs指针参数

2. **Clang Driver集成**
   - 添加`-fcuda-args-profile`编译选项
   - 自动启用profiling pass
   - 无缝集成到CUDA编译流程

### 运行时组件

3. **Runtime Library** (`cuda_profile_runtime.c`)
   - 提供profiling函数实现
   - 智能参数值解析
   - 日志文件管理
   - 自动初始化/清理

### 工作流程

```
CUDA源码 → Clang前端 → LLVM IR → CudaArgsProfile Pass → 
插入profiling调用 → 代码生成 → 链接运行时库 → 可执行文件
```

运行时：
```
程序启动 → CUDA内核调用 → profiling函数执行 → 
参数记录 → 日志输出 → 程序继续执行
```

## 🛠️ 实现步骤

### 1. LLVM Pass实现

**核心文件：**
- `llvm/include/llvm/Transforms/CudaArgsProfile/CudaArgsProfile.h`
- `llvm/lib/Transforms/CudaArgsProfile/CudaArgsProfile.cpp`
- `llvm/lib/Transforms/CudaArgsProfile/CMakeLists.txt`

**关键技术：**
- 使用现代LLVM Pass Manager API
- IR模式匹配识别CUDA API调用
- 类型分析区分标量和指针
- 函数名解析和demangling

### 2. Clang集成

**修改文件：**
- `clang/include/clang/Driver/Options.td` - 添加命令行选项
- `clang/include/clang/Basic/CodeGenOptions.def` - 添加CodeGen选项
- `clang/lib/CodeGen/BackendUtil.cpp` - 集成到编译流程

### 3. 运行时库

**功能模块：**
- 参数值智能解析
- 类型检测（整数/浮点数）
- 日志文件管理
- 内存安全检查

### 4. 构建系统集成

**修改文件：**
- `llvm/lib/Transforms/CMakeLists.txt`
- `llvm/lib/Passes/CMakeLists.txt`
- `llvm/lib/Passes/PassBuilder.cpp`
- `llvm/lib/Passes/PassRegistry.def`

## 🔧 编译安装

### 前置要求

- LLVM/Clang 源码
- CMake 3.13+
- Ninja 构建系统
- CUDA Toolkit
- C++17 编译器

### 编译步骤

1. **配置构建**
```bash
cd llvm-project
mkdir build && cd build

cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=./opt \
  -DLLVM_ENABLE_PROJECTS="clang" \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
  ../llvm
```

2. **构建和安装**
```bash
ninja
ninja install
```

3. **编译运行时库**
```bash
cd ..
gcc -shared -fPIC -o cuda_profile_runtime.so cuda_profile_runtime.c
```

## 📖 使用方法

### 基本用法

```bash
# 编译CUDA程序并启用profiling
./install/bin/clang++ -fcuda-args-profile \
  --cuda-gpu-arch=sm_50 \
  --no-cuda-version-check \
  -L/usr/local/cuda/lib64 -lcudart \
  ./cuda_profile_runtime.so \
  your_cuda_program.cu -o your_program

# 设置库路径并运行
LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH ./your_program
```

### 高级选项

**环境变量：**
- `CUDA_PROFILE_OUTPUT` - 指定输出文件路径（默认：`cuda_profile.log`）

**编译选项：**
- `-fcuda-args-profile` - 启用profiling
- `-fno-cuda-args-profile` - 禁用profiling

### 示例程序

```cpp
// test_cuda_profile.cu
#include <cuda_runtime.h>
#include <iostream>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void scalar_multiply(float* array, float scalar, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        array[idx] *= scalar;
    }
}

int main() {
    const int N = 1000;
    float *d_a, *d_b, *d_c;
    
    // 分配GPU内存
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    
    // 启动内核
    vector_add<<<4, 256>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    
    scalar_multiply<<<8, 128>>>(d_c, 2.5f, N);
    cudaDeviceSynchronize();
    
    // 清理
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}
```

## 🔍 技术细节

### Pass实现原理

1. **函数识别**：在LLVM IR中识别`cudaLaunchKernel()`调用
2. **参数提取**：从调用指令中提取内核函数指针和参数数组
3. **类型分析**：分析内核函数签名，区分标量和指针类型
4. **代码插入**：在调用前插入profiling函数调用
5. **名称解析**：提取并简化内核函数名称

### 运行时库设计

1. **智能类型检测**：
   - 整数范围检测（-1,000,000 到 1,000,000）
   - 浮点数有效性检查
   - 指针地址识别和过滤

2. **内存安全**：
   - 空指针检查
   - 访问边界验证
   - 异常处理

3. **性能优化**：
   - 最小化运行时开销
   - 延迟初始化
   - 缓冲输出

### 局限性

- 只支持标量参数profiling（整数、浮点数）
- 不支持结构体或数组参数
- 需要重新编译LLVM/Clang
- 仅在Linux系统测试

## 🐛 故障排除

### 常见问题

1. **编译错误：`undefined reference to profile_*`**
   - 确保链接了运行时库：`./cuda_profile_runtime.so`
   - 检查`LD_LIBRARY_PATH`设置

2. **Pass未生效**
   - 检查`-fcuda-args-profile`选项是否正确
   - 确认LLVM/Clang正确安装了修改版本

3. **运行时崩溃**
   - 检查CUDA版本兼容性
   - 尝试不同的GPU架构：`--cuda-gpu-arch=sm_60`

4. **空日志文件**
   - 确认程序中有CUDA内核调用
   - 检查文件权限和路径

### 调试技巧

```bash
# 查看编译详细信息
./install/bin/clang++ -v -fcuda-args-profile ...

# 检查Pass是否加载
export LLVM_DEBUG=1
./install/bin/clang++ -fcuda-args-profile ...

# 查看LLVM IR
./install/bin/clang++ -S -emit-llvm -fcuda-args-profile ...
```

## 📄 许可证

本项目基于Apache License 2.0许可证，与LLVM项目保持一致。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个工具！

## 📞 联系

如有问题或建议，请通过GitHub Issues联系。
