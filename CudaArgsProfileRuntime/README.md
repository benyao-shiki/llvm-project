### CudaArgsProfileRuntime

一个用于采集 CUDA 核函数实参与发射配置的轻量运行时库。它通过拦截 CUDA 运行时内存 API 跟踪分配区间，并在每次核函数发射时（由编译器插桩调用）解析参数值，最终写入一个 JSON 文件；程序退出时还会基于这些记录做一次“热门核函数”离线分析。

---

### 工作原理

- 拦截 CUDA 内存 API：通过 `dlsym(RTLD_NEXT, ...)` 拦截 `cudaMalloc/cudaFree/cudaMallocHost/cudaFreeHost/cudaMallocManaged` 等，维护一张内存区表（起始地址、大小、类型：`device|host|managed`）。
- 核函数发射记录：编译器端在每次核函数发射前调用导出符号 `__cuda_profile_kernel_launch(...)`，传入：
  - 核函数名、参数个数、每个实参地址数组 `void** arg_values`
  - 参数类型与布局描述的 JSON 串（含参数名、类型、大小、结构体成员偏移等）
  - `grid`/`block` 维度
  运行时根据该类型描述递归解析参数：
  - 标量：直接读取具体值
  - C 数组如 `T[n]`：逐元素读取值
  - 结构体：按成员偏移读取每个成员的值
  - 指针 `T*`：记录十六进制地址字符串，并结合内存区表补充 `memory_type`、`base_address`、`offset`、以及从该指针到区间末尾的剩余 `size`
- 写文件与并发安全：通过 `atexit` 注册写文件逻辑；支持 `CUDA_ARGS_PROFILE_APPEND=1` 合并已有文件，并使用文件锁避免并发写冲突。
- 退出后处理：写出原始 `kernels` 后，进行离线分析，生成 `hot_kernels` 并与 `kernels` 一起覆盖写回同一 JSON。

---

### JSON 内容结构

- 顶层键：
  - `kernels`: 所有核函数发射的逐次记录数组
  - `hot_kernels`: 退出后离线分析的结果（按核名聚合）

- 每条 `kernels[i]`（一次 kernel launch）包含：
  - `id`: 自增编号（进程内）
  - `name`: 核函数名（来自编译端提供的类型/布局 JSON）
  - `grid`: `[x, y, z]`
  - `block`: `[x, y, z]`
  - `params`: 参数数组。每个参数至少含：
    - `name`: 形参名
    - `type`: 类型字符串，支持如 `int`, `float`, `MyStruct`, `int[3]`, `T*` 等
    - 可能的 `size`: 字节大小（数组/结构体来自类型描述；对指针会被更新为“从该指针到分配区末尾的剩余大小”）
    - 按类型附加字段：
      - 标量：`value` 为具体值（bool/整数/浮点/char）
      - C 数组：`value` 为元素值数组
      - 结构体：`value` 为成员数组；成员项同样含 `name/type/size/value`
      - 指针：
        - `value`: 指针地址的十六进制字符串
        - 若命中已跟踪的分配区，还会包含：
          - `size`: 剩余字节数（从该指针到分配区末尾）
          - `memory_type`: `device|host|managed`
          - `base_address`: 分配区起始地址（十六进制字符串）
          - `offset`: 指针相对分配区起始的字节偏移

- 每条 `hot_kernels[j]`（按核名聚合）包含：
  - `name`: 核函数名
  - `launch_count`: 发射次数
  - 可能包含：
    - `noalias_pointers`: 指针“互不别名”对/项（跨多次发射检测地址区间是否重叠，计算 `noalias_ratio`）
    - `common_scalars`: 稳定出现的常见标量或网格维度（`grid.x/.y/.z`、`block.x/.y/.z`），给出 `value` 与出现比例 `ratio`

---

### 环境变量

- 输出与写入行为：
  - `CUDA_ARGS_PROFILE_JSON_FILE`（必须）: JSON 输出文件路径
  - `CUDA_ARGS_PROFILE_APPEND=1`（可选）: 以追加模式合并写入（带文件锁）
- 离线分析阈值（可选）：
  - `HOT_KERNELS_MIN_LAUNCH_COUNT`（默认 1）
  - `HOT_KERNELS_MIN_NOALIAS_RATIO`（默认 0.6）
  - `HOT_KERNELS_MIN_COMMON_SCALAR_RATIO`（默认 0.6）

---

### 构建

项目使用 CMake，依赖 CUDA 与单文件 `nlohmann::json`（已随库提供 `json.hpp`）。

```bash
mkdir -p /home/yaoben/work/llvm-project/CudaArgsProfileRuntime/build && cd /home/yaoben/work/llvm-project/CudaArgsProfileRuntime/build
cmake ..
make -j
# 生成 libCudaArgsProfileRuntime.so
```

---

### 使用

该库依赖“编译器插桩”在每次核函数发射前调用 `__cuda_profile_kernel_launch(...)` 并传入参数类型/布局 JSON 串。因此：

- 若使用了集成插桩的编译器（例如在 `clang`/`LLVM` 分支中添加了对应支持），只需在运行时让本库优先生效（例如通过 `LD_PRELOAD` 或链接顺序置前）。
- 仅靠拦截内存 API 并不足以记录核函数参数；没有插桩将不会产生 `kernels` 记录。

最简运行示例（假设已具备插桩支持）：

```bash
export CUDA_ARGS_PROFILE_JSON_FILE=/tmp/cuda_args_profile.json
export CUDA_ARGS_PROFILE_APPEND=1   # 可选
LD_PRELOAD=/home/yaoben/work/llvm-project/CudaArgsProfileRuntime/build/libCudaArgsProfileRuntime.so \
  ./your_cuda_app
```

程序结束后，`/tmp/cuda_args_profile.json` 将包含 `kernels` 及聚合后的 `hot_kernels`。

---

### 热门核函数分析说明

- 聚合维度：按核函数名聚合所有发射记录
- 输出：
  - `launch_count`
  - `noalias_pointers`：基于跨次发射指针区间不重叠的比例 `noalias_ratio` ≥ 阈值（默认 0.6）才输出
  - `common_scalars`：跨次发射中占比 ≥ 阈值（默认 0.6）的稳定参数/网格配置

---

### 限制与注意事项

- 必须有编译期插桩提供参数类型/布局 JSON 串，否则无法解析参数值。
- 对指针参数的 `size` 为“从该指针到分配区末尾的剩余字节数”，不是静态类型大小。
- 当前未记录 `cudaMemcpy/cudaMemcpyAsync` 的单次拷贝明细（仅拦截占位）。
- JSON 文件写入使用文件锁，但跨进程大规模并发写入仍建议使用 `CUDA_ARGS_PROFILE_APPEND=1` 并在后处理阶段再汇总。

---

### 精简示例输出

```json
{
  "kernels": [
    {
      "id": 0,
      "name": "myKernel",
      "grid": [80, 1, 1],
      "block": [128, 1, 1],
      "params": [
        {"name": "n", "type": "int", "size": 4, "value": 1024},
        {"name": "alpha", "type": "float", "size": 4, "value": 0.1},
        {"name": "a", "type": "float*", "value": "0x7f...", "size": 4096, "memory_type": "device", "base_address": "0x7f...", "offset": 0}
      ]
    }
  ],
  "hot_kernels": [
    {
      "name": "myKernel",
      "launch_count": 12,
      "noalias_pointers": [
        {"param1_name": "a", "param1_index": 2, "param2_name": "b", "param2_index": 3, "noalias_ratio": 0.92}
      ],
      "common_scalars": [
        {"name": "n", "value": 1024, "ratio": 1.0, "param_index": 0},
        {"name": "grid.x", "value": 80, "ratio": 1.0}
      ]
    }
  ]
}
```

