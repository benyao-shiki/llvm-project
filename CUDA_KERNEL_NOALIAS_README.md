# CUDA Kernel Noalias 优化

本文档描述本仓库中 LLVM/Clang CUDA 工具链的“Kernel Noalias”优化的设计、输入输出格式、启用方式以及前端的动态选择逻辑。内容已与当前实现保持一致，并与 const 优化在风格与命名匹配策略上对齐。

---

## 1. 背景动机

很多 CUDA kernel 接收多个指针实参，这些指针在实际运行中往往互不别名。如果编译器知道这一点，可以更激进地做内存优化（如 LICM、向量化、访存合并等）。Noalias 优化通过“离线 profile + 编译期/前端协作”的方式，在不改变语义的前提下自动选择添加 `noalias` 属性的 kernel 变体，以获得性能收益：

- Profile 收集每次 launch 的指针（含结构体成员路径）的别名标记；
- LLVM pass 基于所有 launches 的统计，结合 `CudaKernelAnalysis` 的指针权重，选择“高收益、且高比例不发生别名”的指针子集，为此克隆 kernel 并在克隆上标注 `noalias`；
- Clang 前端在每个 launch 站点插入一次轻量运行时检查，动态选择“优化 stub”还是“原始 stub”，语义安全。

---

## 2. JSON Profile 输入与结果输出

当前实现不再依赖历史的 `hot_kernels` 摘要，而是只使用逐次 launch 的明细记录 `kernels[]`。每个 kernel 记录包含其一次 launch 的参数展开，其中指针（含结构体成员）带有 `alias` 标识：

- `alias = 0` 表示该 launch 中该指针未与其他指针发生别名（可作为 noalias 候选的积极证据）。
- 未出现或非零表示未知/可能别名。

当参数为结构体或数组时，成员被递归展开。关键在于，每个成员现在通过其**字节偏移量** (`offset`) 而非逻辑索引 (`index`) 来唯一标识。

示例（简化）：

```json
{
  "kernels": [
    {
      "name": "__device_stub__gemm_kernel...",
      "params": [
        { "index": 0, "offset": 0, "type": "int",   "value": 512 },
        { "index": 1, "offset": 8, "type": "float*", "alias": 0 },
        { "index": 2, "offset": 16, "type": "float*", "alias": 0 },
        { "index": 3, "offset": 24, "type": "float*", "alias": 1 }
      ]
    }
  ]
}
```

Pass 选择完成后，会将选择结果写入：

- 以“单文件合并”的方式补写回 profile：键 `cuda_noalias_selected`；或
- 若传入 `-mllvm -cuda-kernel-noalias-selected-out=<file>`，则以数组形式输出到该文件。

输出形如：

```json
{
  "cuda_noalias_selected": [
    {
      "name": "_Z11gemm_kerneliiiffPfS_S_",
      "selected": [
        { "indices": [0, 8], "value": "", "ratio": 1.0 },
        { "indices": [0, 16], "value": "", "ratio": 0.9 }
      ]
    }
  ]
}
```

其中：
- `indices` 为 `[ArgIndex, ByteOffset]` 路径。
- `value` 字段为与 const 统一的占位，noalias 不使用。
- `ratio` 为该条目的单独“无别名比例”，用于日志与调试。

---

## 3. LLVM Pass（CudaKernelNoaliasPass）工作机制

位置：`llvm/lib/Transforms/CudaKernelNoalias/`

启用：

```bash
clang++ ... -mllvm -cuda-kernel-profile=/path/to/profile.json
# 可选：-mllvm -cuda-kernel-noalias-debug -mllvm -cuda-kernel-noalias-selected-out=/tmp/noalias.json
```

流程：
- 解析 `kernels[]`，通过递归地累加 `offset` 字段，将每次 launch 扁平化为“`argIndex.byteOffset` -> alias 标记”的映射；
- 从 `CudaKernelAnalysis` 读取指针权重表，其中每个成员都由其字节偏移量唯一标识；
- 枚举候选指针集合的所有非空子集，基于字节偏移量路径进行匹配和计分；
- 克隆 kernel 得到 `<orig>_noalias`。根据分析结果，采用混合策略添加 noalias 信息：
  - **对于顶层指针参数**（分析结果中 `indices` 路径长度为 1），直接在克隆函数的对应形参上添加 `noalias` 属性。
  - **对于结构体嵌套指针**（`indices` 路径长度 >= 2），则通过 `MDBuilder` 创建独立的别名作用域（alias scope），并为所有相关的内存访问指令（load/store）附加 `!alias.scope` 和 `!noalias` 元数据。
- 复制 `nvvm.annotations` 的 `kernel` 标注，使克隆在设备侧可见；
- 以 `!cuda.noalias.selected` 元数据记录结果（路径为 `[arg, offset]`），并按需写回 JSON。

名称匹配：当前已不再需要复杂的名称规范化逻辑。Pass 直接使用 `device_side_name` (若存在) 或原始 `name` 与 IR 中的函数名进行匹配。

---

## 4. 前端动态选择逻辑（CGCUDARuntime.cpp）

位置：`clang/lib/CodeGen/CGCUDARuntime.cpp`

启用：

```bash
clang++ ... -fcuda-kernel-noalias -mllvm -cuda-kernel-profile=/path/to/profile.json
```

运行时在每个 kernel launch 站点插入：
- 从 `cuda_noalias_selected` 读到的 `[arg, offset]` 列表构造“目标指针集”。
- **获取成员地址**：前端通过 `ASTContext::getASTRecordLayout` 获取结构体的内存布局，然后遍历其字段以找到与元数据中 `offset` 匹配的 `FieldDecl`，从而准确获取成员指针的地址。
- “其他指针集”则为本次调用所有指针实参（顶层）减去目标顶层索引；
- 调用钩子 `check_ptr_sets(int n_targets, void** targets, int n_others, void** others)`：若返回“无别名”，则调用 `__device_stub__foo_noalias<<<...>>>`，否则调用原始 `__device_stub__foo<<<...>>>`。

外部符号：
- `check_ptr_sets` 仅声明不定义，需要在链接/运行时由外部库提供（如基于 nvbit 的实现）。

---

## 5. 构建与运行

(流程不变)

---

## 6. 调试与日志

- `-mllvm -cuda-kernel-noalias-debug`：打印候选集合、子集比例/得分、最终选择以及被跳过原因（无 profile/launches/candidates）。
- `-mllvm -cuda-kernel-noalias-selected-out=/tmp/noalias.json`：将选择结果按数组形式单独输出，便于前端/人工检查。

---

## 7. 与 const 优化的一致性

- **路径表示**：两个优化现在都统一使用 `[ArgIndex, ByteOffset]` 作为参数成员的唯一标识，解决了逻辑索引与物理布局不匹配的问题。
- **Profile 解析**：都采用相同的递归累加 `offset` 的逻辑来处理 profile JSON。
- **前端代码生成**：都使用 `ASTRecordLayout` 来从字节偏移量反向查找 `FieldDecl`，以在运行时获取正确的参数地址/值。

---

## 8. 已知限制与后续工作

- 目前 noalias 的运行时判定在每次 launch 以“集合级别”区分，无更细粒度（未来可扩展为多子集/多变体）。
- 仅对克隆的顶层参数添加 `noalias`，结构体成员的 noalias 语义通过“选择与聚合”体现；如需更细致的 IR 级别刻画可考虑后续扩展。
- 钩子 `check_ptr_sets` 的实现与系统/驱动强相关，需保证与编译期预期一致。 