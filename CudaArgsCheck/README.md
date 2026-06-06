### CudaArgsCheck

A lightweight runtime library for checking CUDA kernel arguments for potential issues like pointer aliasing and constant value mismatches.

---

### How It Works

The library intercepts CUDA Driver API memory management functions (e.g., `cuMemAlloc`, `cuMemFree`) to track allocated memory regions. This is achieved by using `dlsym` to obtain pointers to the original CUDA functions and wrapping them. This approach removes the dependency on `nvbit` for querying memory allocation sizes.

---

### API

The library exports two primary functions:

- `bool check_ptr_sets(int num_targets, void* const* targets, int num_others, void* const* others)`:
  Checks for memory overlap between a set of `targets` pointers and another set of `others` pointers. It returns `true` if any overlap is detected, and `false` if no tracked overlap is found.

- `bool check_const(int n_values, int64_t *expected, int64_t *actual)`:
  Compares two arrays of `int64_t` values and returns `true` if they are identical.

注意：CUDA noalias 动态分流代码使用的是 `!check_ptr_sets(...)`。也就是说，`check_ptr_sets(...) == true` 表示发现重叠/潜在 alias，需要回退原始 kernel；只有返回 `false` 时，noalias 条件才成立。

---

### Build

To build the library, simply run `make` in the `CudaArgsCheck` directory:

```bash
make
```

This will produce the shared library `libcheckkernel.so` in the `CudaArgsCheck/build` directory.

---

### Usage

You can use this library by linking it against your application or by using `LD_PRELOAD` to inject it at runtime.

**Example with LD_PRELOAD:**

```bash
LD_PRELOAD=/path/to/CudaArgsCheck/build/libcheckkernel.so ./your_application
```

When the compiler is built with CUDA kernel const/noalias support, the generated host-side launch code may call these functions before choosing a specialized stub:

- const only: `check_const(...)` passes -> call `<kernel>_const`
- noalias only: `!check_ptr_sets(...)` passes -> call `<kernel>_noalias`
- const + noalias: both checks pass -> call `<kernel>_const_noalias`; otherwise fall back to the best valid single-specialized version or the original kernel
