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

若同时开启 `-fcuda-kernel-const` 与 `-fcuda-kernel-noalias`，noalias pass 还会在已有 `<orig>_const` 的基础上生成 `<orig>_const_noalias`，用于承载“常量传播 + noalias”的组合特化版本。

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

联合开启示例：

```bash
clang++ ... -fcuda-kernel-const -fcuda-kernel-noalias \
  -mllvm -cuda-kernel-const-profile=/path/to/profile.json \
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

联合优化时的组合 clone 命名：`<orig>_const_noalias`

noalias pass 会跳过已经生成的特化版本（`_const`、`_noalias`、`_const_noalias`），避免重复递归克隆；但在处理原始 kernel 时，如果模块中已经存在同名 `<orig>_const`，则额外克隆该 const 版本并对组合 clone 应用同一组选中的 noalias 变换。

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
4. 对返回值取反得到 `NoaliasOK`
5. `NoaliasOK == true` 调 `_noalias` stub；否则调原 stub

这种“在线守卫 + 离线特化”的方式保证语义安全。

同时开启 const/noalias 时，host 端会生成四路分流：

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

其中 `ConstOK = check_const(...)`，`NoaliasOK = !check_ptr_sets(...)`。`check_ptr_sets(...)` 的返回值表示“发现重叠/潜在 alias”，因此需要取反后才表示 noalias 条件成立。

## 8. host/device stub 与符号命名

设备端 kernel clone 与 host 端 stub 均使用相同后缀规则：

- 原始版本：`<orig>`
- 常量传播版本：`<orig>_const`
- noalias 版本：`<orig>_noalias`
- 联合版本：`<orig>_const_noalias`

`CGCUDANV.cpp` 会在生成 CUDA host stub 时额外生成 `_noalias`、`_const` 和 `_const_noalias` stub，并在注册阶段把这些 stub 分别注册到带后缀的 device kernel 名称。注册匹配时 `_const_noalias` 优先于 `_const` 和 `_noalias`，避免组合后缀被误判成单独优化后缀。

## 9. 测试与验证建议

可用 `cuobjdump --dump-elf` 或 `nm` 检查 fatbin/二进制中的 device 符号，确认是否生成了目标 clone。例如联合开启后应能看到：

```text
<orig>
<orig>_const
<orig>_noalias
<orig>_const_noalias
```

运行时验证需要加载检测库：

```bash
LD_PRELOAD=/path/to/CudaArgsCheck/build/libcheckkernel.so ./your_cuda_app
```

若使用 Nsight Compute 采样实际 kernel，需要注意两点：

- 编译的 `--cuda-gpu-arch` 必须匹配实际 GPU，例如 V100 使用 `sm_70`。
- 若 ncu 报 `ERR_NVGPUCTRPERM`，说明当前用户没有访问 NVIDIA performance counter 的权限；此时程序仍可运行，但 ncu 无法输出 per-kernel metrics。

## 10. 目前边界

- 子集搜索是指数复杂度，候选过多时编译成本增大。
- 嵌套路径定位依赖可识别的 IR 访问链。
- 收益取决于 profile 代表性、命中率与检测开销平衡。
- 联合优化复用 const/noalias 两个 pass 各自的选择结果，不额外搜索“常量路径与指针路径”的联合最优子集。
