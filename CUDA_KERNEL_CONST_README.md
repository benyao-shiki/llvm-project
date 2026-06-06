# CUDA Kernel Constant Propagation（当前实现说明）

本文档对应当前 `cudakernel_alias` 分支实现，描述 `CudaKernelConstPass` 的真实行为、选项和与前端协作方式。

## 1. 总体流程

1. 运行时 profile 记录 `kernels[]` 每次 launch 的参数值、结构体成员值、`grid/block`。
2. `CudaKernelConstPass` 读取 profile + `CudaKernelAnalysis` 权重，选出收益最高的参数子集。
3. Pass 克隆 kernel 为 `<orig>_const`，仅在 clone 内进行常量替换。
4. host 端根据 `cuda_const_selected` 做运行时比较，命中时调用 `_const` stub，否则回退原 stub。

若同时开启 `-fcuda-kernel-const` 与 `-fcuda-kernel-noalias`，当前实现还会生成联合版本 `<orig>_const_noalias`。联合版本的设备端构造方式是：先由 const pass 生成 `<orig>_const`，再由 noalias pass 在 `<orig>_const` 基础上继续克隆并施加 noalias 变换。

## 2. Profile 输入

Pass 读取 JSON 根下的 `kernels[]`，不依赖旧 `hot_kernels`。

每条 launch 记录按路径扁平化后参与统计：

- 顶层标量：`"argIndex"` / `"argIndex.0"`
- 结构体成员标量：`"argIndex.byteOffset"`
- 维度参数：
  - `grid.x/y/z` 映射到保留索引
  - `block.x/y/z` 映射到保留索引

kernel 名优先使用 `device_side_name`，缺失时使用 `name`。

## 3. 启用方式与关键选项

```bash
clang++ ... -fcuda-kernel-const \
  -mllvm -cuda-kernel-const-profile=/path/to/profile.json
```

联合开启示例：

```bash
clang++ ... -fcuda-kernel-const -fcuda-kernel-noalias \
  -mllvm -cuda-kernel-const-profile=/path/to/profile.json \
  -mllvm -cuda-kernel-profile=/path/to/profile.json
```

可选项：

- `-mllvm -const-min-weight=<N>`
  - 默认 `0`
  - 只有 `weight > N` 的分析路径才进入候选集（严格大于）
- `-mllvm -cuda-kernel-const-debug`
  - 打印子集评分与最终选择
- `-mllvm -cuda-kernel-const-selected-out=/path/to/out.json`
  - 将选择结果写到独立文件

profile 路径优先级：

1. `-cuda-kernel-const-profile`
2. 环境变量 `CUDA_KERNEL_PROFILE`

## 4. 候选选择与打分

候选来自 `CudaKernelAnalysis::ScalarWeights`，再按 `const-min-weight` 过滤。

对任意非空子集 `S`：

- 统计该子集“联合取值组合”在 launch 中的众数比例 `Ratio(S)`
- `SumWeight(S) = Σ weight(p), p∈S`
- `Score_const(S) = Ratio(S) * SumWeight(S)`

并列时，优先选择元素个数更多的子集。

## 5. clone 内常量传播范围

clone 命名：`<orig>_const`

联合优化时，还会存在组合 clone：`<orig>_const_noalias`。该 clone 继承 `_const` 中已经完成的常量替换，再叠加 noalias pass 选择出的指针属性或 alias metadata。

支持三类路径：

1. 顶层标量参数
   - 直接替换参数 uses
2. 结构体成员标量（按字节偏移）
   - 匹配对应 `GEP(+bitcast)+load` 链并替换 load 结果
3. `grid/block` 维度
   - 将 `llvm.nvvm.read.ptx.sreg.*` 调用替换为常量并删除调用

不做的事情：

- 不做“普通指针地址常量化”
- 不在原函数上改写，只改 clone

## 6. 选择结果输出（当前真实接口）

当前实现通过 JSON 输出选择结果，键名为 `cuda_const_selected`。

支持两种形态（读写都兼容）：

1. 对象形态：`{ "kernel": [items...] }`
2. 数组形态：`[{"name":"kernel","selected":[...]}]`

每个 item 典型字段：

- `indices`: 参数路径（如 `[2,0]` 或 `[1,16]`）
- `value`: 目标常量值（字符串）
- `ratio`: 单项命中比例
- `type`: 推断的标量类型（如 `i32/float`）

说明：当前实现不依赖模块级 `!cuda.const.selected` 命名元数据。

## 7. 与前端动态分流协作

host 端（`CGCUDARuntime.cpp`）读取 `cuda_const_selected`，在 launch 点生成类似逻辑：

```c
if (check_const(...expected..., ...actual...))
  __device_stub__foo_const<<<...>>>();
else
  __device_stub__foo<<<...>>>();
```

其中：

- 结构体成员实际值按 `indices` 路径从 AST/布局中提取
- 命中才走 `_const`，否则始终回退原始路径，语义安全

同时开启 const/noalias 时，host 端会同时读取 `cuda_const_selected` 与 `cuda_noalias_selected`，并在同一个 kernel launch 处生成四路分流：

```c
if (ConstOK) {
  if (NoaliasOK)
    __device_stub__foo_const_noalias<<<...>>>();
  else
    __device_stub__foo_const<<<...>>>();
} else {
  if (NoaliasOK)
    __device_stub__foo_noalias<<<...>>>();
  else
    __device_stub__foo<<<...>>>();
}
```

其中 `ConstOK` 来自 `check_const(...)`，`NoaliasOK` 来自 `!check_ptr_sets(...)`。也就是说，只有常量条件和指针不重叠条件同时满足时，才会调用 `_const_noalias` 版本。

## 8. 与 noalias 联合优化的设备端顺序

后端 pass 注册顺序为：

1. `CudaKernelConstPass`
2. `CudaKernelNoaliasPass`

这一顺序是联合优化成立的关键：`CudaKernelConstPass` 先生成 `<orig>_const`；随后 `CudaKernelNoaliasPass` 除了从原函数生成 `<orig>_noalias`，还会检测是否存在 `<orig>_const`，若存在则继续生成 `<orig>_const_noalias`。

## 9. 目前边界

- 候选子集是指数枚举，参数过多时编译开销会上升。
- 结构体成员常量替换依赖可识别的 IR 访问形态（主要是常量偏移 GEP 链）。
- 收益高度依赖 profile 的代表性与在线命中率。
- 联合优化并不重新做“const 与 noalias 的联合候选搜索”，而是复用两个 pass 各自已经选出的候选集合，再在设备端和 host 端组合这些选择。
