# CudaArgsProfileRuntime

`CudaArgsProfileRuntime` 是 `kernel_profile` 分支的运行时采样库。完整的系统设计、前端插桩与 JSON 格式说明见仓库根目录 `README_CudaArgsProfile.md`；本文档集中说明动态库自身的构建、运行方式与实现边界。

## 工作方式

编译器生成的 host CUDA stub 会在 kernel launch 之前调用：

```cpp
void __cuda_profile_kernel_launch(const char *kernel_name,
                                  const char *device_side_name,
                                  int arg_count,
                                  void **arg_values,
                                  const char *arg_info_json_str,
                                  void *grid_dim,
                                  void *block_dim);
```

动态库完成两类工作：

1. Hook `cudaMalloc/cudaFree`、`cudaMallocHost/cudaFreeHost`、`cudaMallocManaged`，维护 `device|host|managed` 内存区间。
2. 在回调中结合编译器提供的参数布局 JSON，读取本次 launch 的参数值与 `grid/block` 配置。

`cudaMemcpy/cudaMemcpyAsync` 当前已包装并转发到原 runtime，但未写入拷贝记录。

## 参数采样行为

- 标量：读取实际值；不支持的类型写为 `"unsupported_scalar_type"`。
- C 数组：匹配 `T[n]` 后按标量元素逐项读取；复杂元素数组没有递归解析实现。
- 结构体：按成员字节 `offset` 递归读取，包含嵌套结构体。
- 指针：记录地址字符串；若位于已跟踪的分配区，追加 `memory_type`、`base_address`、`offset` 与从当前地址到分配末尾的 `size`。
- 结构体中的指针成员：递归过程中同样按指针语义解析。

指针值只用于地址和区间判断，库不会读取指针指向的 device 数据。

## Alias 与离线汇总

单次 launch 中，回调直接给顶层指针写入 `alias`：可比较的区间重叠则为 `1`，否则为 `0`。未命中内存区间表的顶层指针可能仍保留 `alias = 0`，因此严格使用此信息时应确保指针来自被 hook 的分配 API。

程序正常退出时，库先写出 `kernels[]`，再运行 `process_profile_data()`：

- 按 `launch["name"]` 聚合，而不是按 `device_side_name`。
- 输出 `hot_kernels[].launch_count`。
- 输出超过阈值的 `common_scalars`，包含标量、数组整体值和 `grid/block` 维度。
- 递归发现顶层或结构体成员指针，基于区间不重叠比例输出 `noalias_pointers`。

## 环境变量

| 变量 | 默认值 | 作用 |
|---|---:|---|
| `CUDA_ARGS_PROFILE_JSON_FILE` | 无 | 必须设置，指定 JSON 输出路径 |
| `CUDA_ARGS_PROFILE_APPEND` | 未启用 | 设为 `1` 时合并已有 `kernels[]` |
| `HOT_KERNELS_MIN_LAUNCH_COUNT` | `1` | 生成 hot kernel 的最少 launch 次数 |
| `HOT_KERNELS_MIN_NOALIAS_RATIO` | `0.6` | 输出 noalias 指针对的最小比例 |
| `HOT_KERNELS_MIN_COMMON_SCALAR_RATIO` | `0.6` | 输出常见值的最小比例 |

写文件阶段使用文件锁。追加多个进程的采样记录时，`kernels[].id` 不是全局唯一编号。

## 构建

```bash
cd /home/yaoben/bishe/llvm-project/CudaArgsProfileRuntime
cmake -S . -B build
cmake --build build -j
```

输出动态库：

```text
CudaArgsProfileRuntime/build/libCudaArgsProfileRuntime.so
```

## 使用

应用必须由包含 host stub 插桩实现的 Clang 编译。当前分支虽然声明了 `-fcuda-args-profile`，但 `emitDeviceStubBodyNew` 中的回调生成尚未受该开关条件控制；新式 stub 路径下会生成回调调用。

运行示例：

```bash
export CUDA_ARGS_PROFILE_JSON_FILE=/tmp/cuda_args_profile.json
export CUDA_ARGS_PROFILE_APPEND=1
LD_PRELOAD=/home/yaoben/bishe/llvm-project/CudaArgsProfileRuntime/build/libCudaArgsProfileRuntime.so ./your_cuda_app
```

程序正常结束后，输出文件包含：

- `kernels[]`：逐次 launch 的参数值、结构信息、地址信息及发射配置。
- `hot_kernels[]`：按 kernel 名汇总的常见值与 noalias 候选信息。

## 已知限制

1. 写出依赖 `atexit`，异常退出可能丢失未落盘采样。
2. 复杂元素数组当前不会像结构体参数一样递归展开。
3. 离线 noalias 统计假设各次 launch 中可识别指针的形态保持一致。
4. `hot_kernels` 的成员标识主要用于摘要展示；保留完整结构与偏移的原始输入是 `kernels[]`。
