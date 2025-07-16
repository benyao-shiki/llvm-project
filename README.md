# CudaArgsProfile LLVM Pass

## 概述

CudaArgsProfile是一个LLVM Pass，专门用于分析和记录CUDA kernel启动时的参数信息。它能够自动插入profiling代码，收集kernel参数的详细信息，并生成JSON格式的分析报告。

## 核心功能

### 主要特性

1. **自动参数分析**: 自动分析CUDA kernel函数的参数类型和结构
2. **结构体参数重建**: 利用调试信息重建结构体成员的可读形式
3. **结构体分割处理**: 智能处理被LLVM IR分割的结构体参数
4. **详细的JSON报告**: 生成包含参数值、类型、大小等详细信息的JSON报告
5. **支持多种数据类型**: 标量、指针、结构体等各种CUDA参数类型

### 支持的参数类型

- **标量类型**: int8, int16, int32, int64, float, double
- **指针类型**: 设备指针、主机指针
- **结构体类型**: 普通结构体和被分割的结构体
- **复合类型**: 嵌套结构体（通过调试信息解析）

## 技术背景

### 结构体分割问题

在x86-64 ABI中，超过8字节的结构体参数会被编译器自动分割成多个8字节的部分传递。这导致了一个关键问题：

```c
// 源代码
struct Data16 {
    int value;        // 4字节
    float rate;       // 4字节
    double precision; // 8字节
};

void kernel(struct Data16 data, int other);
```

```llvm
; LLVM IR层面
define void @kernel(i64 %0, double %1, i32 %other)
; 16字节结构体被分割为: i64(前8字节) + double(后8字节)
```

**问题**: 传统的参数profiling方法假设`args[i]`直接对应第i个原始参数，但实际上：
- `args[0]` = 结构体的前8字节 
- `args[1]` = 结构体的后8字节
- `args[2]` = other参数

### 解决方案

本项目通过以下技术手段解决了结构体分割问题：

1. **调试信息分析**: 解析DWARF调试信息获取结构体成员详情
2. **参数映射**: 建立原始参数到IR参数的映射关系
3. **内存重建**: 从分割的parts重建完整的结构体数据
4. **智能识别**: 自动识别哪些参数需要分割处理

## 实现架构

### 组件结构

```
CudaArgsProfile/
├── CudaArgsProfile.cpp          # 主要的Pass实现
├── CudaArgsProfileRuntime.c     # 运行时库
└── README.md                    # 本文档
```

### 核心数据结构

#### 1. ArgMappingInfo
```cpp
struct ArgMappingInfo {
  std::string originalName;           // 原始参数名
  ParamTypeInfo originalParam;        // 原始参数信息
  std::vector<unsigned> argsIndices;  // 在args数组中的索引
  std::vector<Type*> splitTypes;      // 分割后的IR类型
  bool isSplit;                       // 是否被分割
};
```

#### 2. ParamTypeInfo
```cpp
struct ParamTypeInfo {
  enum Type { SCALAR_INT8, SCALAR_INT16, SCALAR_INT32, SCALAR_INT64,
              SCALAR_FLOAT, SCALAR_DOUBLE, POINTER, STRUCT };
  Type type;
  size_t size;
  size_t offset;
  std::string name;
  std::vector<ParamTypeInfo> members;  // 结构体成员
};
```

#### 3. MemberInfo (Runtime)
```c
typedef struct {
  char name[64];
  int type;
  int offset;
  int size;
} MemberInfo;
```

### 工作流程

1. **Pass初始化**: 创建运行时函数声明和全局变量
2. **CUDA调用识别**: 识别CUDA kernel启动调用
3. **参数分析**: 分析kernel函数的参数类型和结构
4. **结构体分割分析**: 通过调试信息判断参数是否被分割
5. **代码插入**: 在kernel启动前后插入profiling代码
6. **运行时收集**: 运行时库收集参数值并重建结构体
7. **JSON输出**: 生成详细的分析报告

## 技术细节

### 1. 调试信息解析

```cpp
ArgMappingInfo CudaArgsProfileImpl::analyzeParameterDebugInfo(
    DIType *ParamType, const std::vector<Type*> &IRArgTypes, 
    unsigned &currentIRArgIndex) {
  
  ArgMappingInfo mapping;
  mapping.originalParam = analyzeDebugType(ParamType);
  
  // 判断是否需要分割
  if (mapping.originalParam.size > 8) {
    mapping.isSplit = true;
    // 计算分割后的IR参数索引
    size_t remainingSize = mapping.originalParam.size;
    while (remainingSize > 0 && currentIRArgIndex < IRArgTypes.size()) {
      mapping.argsIndices.push_back(currentIRArgIndex);
      mapping.splitTypes.push_back(IRArgTypes[currentIRArgIndex]);
      remainingSize -= 8;  // 每个分割部分8字节
      currentIRArgIndex++;
    }
  } else {
    mapping.isSplit = false;
    mapping.argsIndices.push_back(currentIRArgIndex);
    mapping.splitTypes.push_back(IRArgTypes[currentIRArgIndex]);
    currentIRArgIndex++;
  }
  
  return mapping;
}
```

### 2. 结构体重建

```cpp
void CudaArgsProfileImpl::handleSplitStructArgument(
    IRBuilder<> &Builder, Value *Args, 
    const ArgMappingInfo &mapping, unsigned paramIndex) {
  
  // 获取完整的结构体数据
  Value *StructArgPtr = Builder.CreateGEP(VoidPtrTy, Args, 
                                         Builder.getInt32(paramIndex));
  Value *StructDataPtr = Builder.CreateLoad(VoidPtrTy, StructArgPtr);
  
  // 创建8字节对齐的分割部分
  size_t numParts = (mapping.originalParam.size + 7) / 8;
  ArrayType *PtrArrayType = ArrayType::get(VoidPtrTy, numParts);
  Value *SplitParts = Builder.CreateAlloca(PtrArrayType);
  
  // 填充分割部分
  for (size_t i = 0; i < numParts; ++i) {
    Value *OffsetValue = Builder.getInt32(i * 8);
    Value *PartPtr = Builder.CreateGEP(Type::getInt8Ty(*Context), 
                                      StructDataPtr, OffsetValue);
    // ... 存储到split_parts数组
  }
  
  // 调用运行时函数
  Builder.CreateCall(ProfileSplitStructFunc, {
    Builder.getInt32(paramIndex), StructNamePtr, 
    SplitPartsPtr, Builder.getInt32(numParts), MemberInfoPtr
  });
}
```

### 3. 运行时成员值提取

```c
void __cuda_profile_split_struct(int index, const char* struct_name, 
                                void** split_parts, int num_parts, 
                                const char* member_info) {
  // 解析成员信息
  // 格式: "member_count:name1:type1:offset1:size1:name2:type2:offset2:size2:..."
  
  char reconstructed[512] = "{";
  
  for (int i = 0; i < member_count; i++) {
    MemberInfo* member = &members[i];
    
    // 根据偏移量找到对应的split part
    int current_offset = 0;
    for (int j = 0; j < num_parts; j++) {
      if (member->offset >= current_offset && 
          member->offset < current_offset + 8) {
        
        // 计算part内偏移
        int part_offset = member->offset - current_offset;
        uint8_t* part_data = (uint8_t*)split_parts[j];
        
        // 根据类型提取值
        if (member->type == SCALAR_INT32) {
          uint32_t val = *(uint32_t*)(part_data + part_offset);
          sprintf(member_value, "%u", val);
        } else if (member->type == SCALAR_FLOAT) {
          float val = *(float*)(part_data + part_offset);
          sprintf(member_value, "%f", val);
        }
        // ... 其他类型处理
        break;
      }
      current_offset += 8;
    }
    
    sprintf(reconstructed + strlen(reconstructed), "%s: %s", 
            member->name, member_value);
  }
  
  strcat(reconstructed, "}");
}
```

### 4. JSON输出格式

```json
{
  "total_kernels": 3,
  "kernels": [
    {
      "id": 0,
      "name": "kernel_function_name",
      "grid": [1,1,1],
      "block": [1,1,1],
      "params": [
        {
          "type": "split_struct",
          "name": "Data16",
          "size": 16,
          "reconstructed_value": "{value: 123, rate: 2.718000, precision: 3.141590}",
          "members": [
            {"name": "value", "type": 2, "offset": 0, "size": 4},
            {"name": "rate", "type": 4, "offset": 4, "size": 4},
            {"name": "precision", "type": 5, "offset": 8, "size": 8}
          ],
          "split_parts": [
            {"index": 0, "size": 8, "value": "..."},
            {"index": 1, "size": 8, "value": "..."}
          ]
        }
      ]
    }
  ]
}
```

## 编译和使用

### 1. 编译Pass

```bash
cd llvm-project
mkdir build_profile
cd build_profile
cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang" ../llvm
ninja
```

### 2. 编译运行时库

```bash
gcc -c CudaArgsProfileRuntime.c -o CudaArgsProfileRuntime.o
```

### 3. 使用Pass

```bash
build_profile/bin/clang++ -fcuda-args-profile -g -O0 \
    your_cuda_program.cu CudaArgsProfileRuntime.o \
    -o your_program \
    --cuda-gpu-arch=sm_80 \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lcudart
```

### 4. 运行和分析

```bash
./your_program
# 生成 cuda_profile.json
```

## 测试示例

### 测试代码

```c
struct Data16 {
    int value;
    float rate;
    double precision;
};

__global__ void test_kernel(struct Data16 data, int other) {
    printf("value=%d, rate=%f, precision=%f, other=%d\n", 
           data.value, data.rate, data.precision, other);
}

int main() {
    struct Data16 data = {123, 2.718f, 3.14159};
    test_kernel<<<1, 1>>>(data, 456);
    cudaDeviceSynchronize();
    return 0;
}
```

### 输出结果

```json
{
  "kernels": [
    {
      "params": [
        {
          "type": "split_struct",
          "name": "Data16", 
          "reconstructed_value": "{value: 123, rate: 2.718000, precision: 3.141590}",
          "members": [
            {"name": "value", "type": 2, "offset": 0, "size": 4},
            {"name": "rate", "type": 4, "offset": 4, "size": 4},
            {"name": "precision", "type": 5, "offset": 8, "size": 8}
          ]
        }
      ]
    }
  ]
}
```

## 优势和应用

### 技术优势

1. **准确性**: 正确处理结构体参数分割问题
2. **完整性**: 支持所有常见的CUDA参数类型
3. **可读性**: 生成人类可读的结构体成员信息
4. **自动化**: 无需手动标注，自动分析参数结构
5. **性能**: 最小化运行时开销

### 应用场景

1. **性能分析**: 分析kernel参数传递的性能影响
2. **调试辅助**: 帮助理解kernel接收的实际参数值
3. **优化指导**: 识别参数传递的瓶颈
4. **代码审查**: 验证参数传递的正确性

## 技术限制

1. **调试信息依赖**: 需要编译时包含调试信息(-g)
2. **DWARF格式**: 依赖标准的DWARF调试信息格式
3. **平台支持**: 主要针对x86-64和CUDA平台
4. **复杂类型**: 对于非常复杂的嵌套结构体可能需要改进

## 未来改进方向

1. **更多数据类型**: 支持数组、联合体等复杂类型
2. **性能优化**: 进一步减少运行时开销
3. **可视化**: 提供图形化的分析界面
4. **扩展平台**: 支持更多GPU架构和平台
5. **深度分析**: 提供更深入的性能分析功能

## 贡献指南

欢迎提交issue和pull request。在提交代码前，请确保：

1. 代码符合LLVM编码规范
2. 添加适当的测试用例
3. 更新相关文档
4. 通过所有现有测试

## 许可证

本项目遵循Apache License 2.0，与LLVM项目保持一致。

## 联系方式

如有问题或建议，请通过GitHub Issues联系。
