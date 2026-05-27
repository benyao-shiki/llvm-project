# CUDA Kernel Noalias（当前实现说明）

本文档对应当前 `cudakernel_alias` 分支实现，描述 `CudaKernelNoaliasPass` 与前端分流的真实行为。

## 1. 总体流程

1. runtime profile 在 `kernels[]` 里记录每次 launch 的指针 alias 标记（含结构体成员路径）。
2. `CudaKernelNoaliasPass` 读取 profile + `CudaKernelAnalysis` 指针权重，选出最佳 noalias 子集。
3. Pass 克隆 kernel 为 `<orig>_noalias`。
4. 在 clone 上：
   - 顶层指针参数加 `noalias` 参数属性；
   - 嵌套指针访问加 `alias.scope/noalias` 元数据。
5. host 端运行时检测通过则调 `_noalias` stub，否则走原 stub。

## 2. Profile 输入与路径

Pass 读取 JSON `kernels[]`，每次 launch 扁平化为 `path -> aliasFlag`：

- 顶层指针：`"argIndex"` / `"argIndex.0"`
- 结构体内指针：`"argIndex.byteOffset"`

`aliasFlag == 0` 视为该路径在该次 launch 满足“无别名证据”。

kernel 名优先读取 `device_side_name`，缺失时使用 `name`。

## 3. 启用方式与关键选项

```bash
clang++ ... -fcuda-kernel-noalias \
  -mllvm -cuda-kernel-profile=/path/to/profile.json
```

可选项：

- `-mllvm -noalias-min-weight=<N>`
  - 默认 `0`
  - 仅 `weight > N` 的指针路径进入候选（严格大于）
- `-mllvm -cuda-kernel-noalias-debug`
  - 打印候选、子集评分、最佳选择
- `-mllvm -cuda-kernel-noalias-selected-out=/path/to/out.json`
  - 将选择结果写到独立文件

profile 路径优先级：

1. `-cuda-kernel-profile`
2. 环境变量 `CUDA_KERNEL_PROFILE`

## 4. 候选选择与打分

候选来自 `CudaKernelAnalysis::PointerWeights`，再按 `noalias-min-weight` 过滤。

对任意非空子集 `S`：

- `Ratio_noalias(S) = 满足子集内所有路径 alias==0 的 launch 数 / 总 launch 数`
- `SumWeight(S) = Σ weight(p), p∈S`
- `Score_noalias(S) = Ratio_noalias(S) * SumWeight(S)`

并列时，优先更大的子集。

## 5. clone 变换细节

clone 命名：`<orig>_noalias`

### 5.1 顶层指针

当路径对应顶层指针参数时，直接给该参数添加 `Attribute::NoAlias`。

### 5.2 结构体嵌套指针

当路径对应嵌套成员指针时：

1. 定位指针值（按 `arg + byteOffset` 追踪）
2. 为每个目标指针创建独立 alias scope（同一 domain）
3. 找到相关内存访问（`load/store`，以及当前实现里被跟踪到的 call）
4. 在访问指令上附加：
   - `!alias.scope`
   - `!noalias`

因此，不是“仅顶层参数生效”，嵌套路径也会在 IR 上体现 noalias 约束。

## 6. 选择结果输出（当前真实接口）

结果写入 JSON 键 `cuda_noalias_selected`，读写兼容两种形态：

1. 对象形态：`{ "kernel": [items...] }`
2. 数组形态：`[{"name":"kernel","selected":[...]}]`

每个 item 主要字段：

- `indices`: 参数路径（如 `[1,24]`）
- `ratio`: 单项无别名比例
- `value`: 占位字段（通常为空字符串）

说明：当前实现不依赖模块级 `!cuda.noalias.selected` 命名元数据。

## 7. 与前端动态分流协作

host 端（`CGCUDARuntime.cpp`）读取 `cuda_noalias_selected`，launch 点执行：

1. 构造目标指针集合（selected 路径）
2. 构造其余指针集合
3. 调用 `check_ptr_sets(...)`
4. 返回 true 调 `_noalias` stub；否则调原 stub

这种“在线守卫 + 离线特化”的方式保证语义安全。

## 8. 目前边界

- 子集搜索是指数复杂度，候选过多时编译成本增大。
- 嵌套路径定位依赖可识别的 IR 访问链。
- 收益取决于 profile 代表性、命中率与检测开销平衡。
