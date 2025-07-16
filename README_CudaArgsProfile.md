# CUDA Arguments Profiling Pass

这个LLVM pass可以在编译时插桩CUDA程序，用于分析kernel启动时的参数信息，包括标量值、指针地址和结构体成员。

## 功能特性

- **参数类型分析**: 支持标量类型（int8/16/32/64, float, double）、指针类型和结构体类型
- **结构体成员解析**: 能够解析一层深度的结构体成员（结构体嵌套会标记为struct type）
- **JSON格式输出**: 生成类似nvbit工具的JSON格式分析结果
- **统计信息**: 提供kernel调用统计、常见参数值分析等
- **指针处理**: 只记录指针地址，不尝试读取device内存内容

## 编译方法

### 1. 编译LLVM和pass

```bash
# 在LLVM项目根目录
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_PROJECTS="clang" \
      -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
      ../llvm
make -j$(nproc)
```

### 2. 编译运行时库

```bash
# 编译运行时库
cd llvm/lib/Transforms/CudaArgsProfile
g++ -c -fPIC CudaArgsProfileRuntime.cpp -o CudaArgsProfileRuntime.o
ar rcs libCudaArgsProfileRuntime.a CudaArgsProfileRuntime.o
```

## 使用方法

### 1. 编译CUDA程序并插桩

```bash
# 使用修改过的clang编译，启用fcuda-args-profile选项
clang++ -fcuda-args-profile test_cuda_profile.cu -o test_cuda_profile \
        -L. -lCudaArgsProfileRuntime -lcudart
```

### 2. 运行程序

```bash
# 运行程序，会自动生成cuda_profile.json
./test_cuda_profile
```

### 3. 查看结果

程序运行后会生成`cuda_profile.json`文件，格式如下：

```json
{
  "total_kernels": 1,
  "hot_kernels": [
    {
      "name": "test_kernel",
      "launches": 1,
      "common_scalars": [
        {"arg": 0, "value": "42", "ratio": 1.00},
        {"arg": 1, "value": "1078530011", "ratio": 1.00}
      ],
      "common_dims": [
        {"dim": "gridDim.x", "value": 32, "ratio": 1.00},
        {"dim": "gridDim.y", "value": 32, "ratio": 1.00},
        {"dim": "gridDim.z", "value": 1, "ratio": 1.00},
        {"dim": "blockDim.x", "value": 32, "ratio": 1.00},
        {"dim": "blockDim.y", "value": 32, "ratio": 1.00},
        {"dim": "blockDim.z", "value": 1, "ratio": 1.00}
      ],
      "noalias_pointers": [
        {"arg": 3, "ratio": 1}
      ]
    }
  ],
  "kernels": [
    {
      "id": 0,
      "name": "test_kernel",
      "grid": [32,32,1],
      "block": [32,32,1],
      "shmem_static": 0,
      "shmem_dynamic": 0,
      "registers": 32,
      "params": [
        {"size": 4, "value": "42"},
        {"size": 4, "value": "1078530011"},
        {"size": 16, "value": "struct_0"},
        {"size": 8, "value": "0x7f1234567890"}
      ]
    }
  ]
}
```

## 输出格式说明

### hot_kernels部分
- `name`: kernel函数名
- `launches`: 调用次数
- `common_scalars`: 常见标量参数值及其出现比率
- `common_dims`: 常见的grid/block维度配置
- `noalias_pointers`: 指针参数的索引信息

### kernels部分
- `id`: kernel调用的唯一ID
- `name`: kernel函数名
- `grid/block`: 启动配置
- `params`: 详细的参数信息
  - `size`: 参数大小（字节）
  - `value`: 参数值（标量显示实际值，指针显示地址，结构体显示类型）

## 参数类型处理

1. **标量类型**: 直接显示数值，float类型会转换为十六进制表示
2. **指针类型**: 显示十六进制地址
3. **结构体类型**: 显示结构体名称，并递归分析一层成员
4. **未知类型**: 按照原始字节处理

## 限制和注意事项

1. **结构体深度**: 目前只支持一层结构体成员解析
2. **指针内容**: 不会尝试读取device指针指向的内容
3. **性能影响**: 插桩会带来一定的运行时开销
4. **兼容性**: 需要修改过的clang支持`-fcuda-args-profile`选项

## 示例程序

参考`test_cuda_profile.cu`中的示例，展示了如何使用各种参数类型：

```cpp
struct MyStruct {
    int a;
    float b;
    double c;
};

__global__ void test_kernel(int scalar_arg, float float_arg, 
                           MyStruct struct_arg, int* ptr_arg) {
    // kernel实现
}
```

这个pass能够准确分析这些参数的类型和值，生成详细的profiling报告。 