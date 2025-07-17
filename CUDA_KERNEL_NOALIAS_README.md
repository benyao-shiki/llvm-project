# CUDA Kernel **Noalias** Optimization

This document explains the design, build-time switches and run-time requirements for the *Kernel Noalias* optimisation implemented in the LLVM/Clang CUDA tool-chain contained in this repository.

---

## 1. Motivation

Many CUDA kernels receive several pointer parameters that are guaranteed **not** to alias each other.  If the compiler is aware of this property it can perform more aggressive memory-optimisations (LICM, vectorisation, memory coalescing, …).  Unfortunately the information is seldom available at compile time.

The *Kernel Noalias* pipeline bridges that gap:

1.  **profilers** a tool based on nvbit, which identify kernels and pointer arguments that never alias in practice and emit a JSON profile.
2.  **LLVM pass** (`CudaKernelNoaliasPass`) clones each hot kernel, annotates the selected parameters with the `noalias` attribute and keeps the original version untouched.
3.  **Clang front-end** (`CGCUDARuntime`) inserts a small run-time check in every launch site that dynamically chooses between the specialised *noalias* stub and the normal stub.
4.  A hook func (`check_ptr_sets`) provides the alias ptr check at run time.

The result is a *fully automatic*, profile-guided optimisation that yields large speed-ups while being 100 % safe – when aliasing is detected we simply fall back to the unoptimised kernel.

---

## 2. JSON profile format

The optimiser consumes a single JSON file (e.g. `profile_test.json`).  A reduced example:

```json
      "name": "_Z11gemm_kerneliiiffPfS_S_",
      "demangled_name": "gemm_kernel(int, int, int, float, float, float*, float*, float*)",
      "launches": 1,
      "common_scalars": [
        {
          "arg": 0,
          "value": "512",
          "ratio": 1
        }
      ],
      "common_dims": [
        {
          "dim": "gridDim.x",
          "value": 16,
          "ratio": 1
        }
      ],
      "noalias_pointers": [
        {
          "arg": 5,
          "ratio": 1
        },
        {
          "arg": 6,
          "ratio": 1
        },
        {
          "arg": 7,
          "ratio": 1
        }
      ]
```

Keys:

*   `hot_kernels[]` – kernels that should be specialised.
*   `name` – *device* side mangled name (same as shown in PTX / `nvcc --ptxas-options=-v`).
*   `noalias_pointers[]` – list of pointer arguments (zero-based) that are known never to alias **any** other pointer argument of the same launch.

If either the kernel is absent from the array or its list is empty, that kernel is left untouched.

---

## 3. LLVM Pass – `CudaKernelNoaliasPass`

Location: `llvm/lib/Transforms/CudaKernelNoalias/`.

Activation:

```bash
clang++ -mllvm -cuda-kernel-profile=/path/to/profile_test.json …
```

Behaviour:

1.  Scans every function with calling convention `ptx_kernel` (or with `nvvm.annotations` == "kernel").
2.  Looks up the function name in the parsed profile map.
3.  If at least one pointer index is listed:
    *   Clones the function with name `<orig>_noalias`.
    *   Adds attribute `noalias` to the specified parameters in the clone **only**.
    *   Copies the original `nvvm.annotations` entry so that the clone is also visible to the GPU driver.


---

## 4. Clang Front-End Support (`CGCUDARuntime.cpp`)

Compile-time switch:

```
–fcuda-kernel-noalias   # sets CodeGenOpts.CudaKernelNoalias
```

(the flag name is indicative – use the one wired in your local driver).

### 4.1  Profile loading

`loadCudaKernelProfile()` is executed on first use and re-uses **the same JSON file** as the LLVM pass.  The path is resolved in the following order:

1.  Environment variable `CUDA_KERNEL_PROFILE`.
2.  Last occurrence of the backend option `-cuda-kernel-profile=<file>`.

### 4.2  Launch-site transformation

At every host-side kernel call Clang now emits:

```
if (!TargetPtrs.empty() &&
    !check_ptr_sets(TargetPtrs.size(), TargetPtrs, OtherPtrs.size(), OtherPtrs))
  __device_stub__foo_noalias<<<…>>>();   // no alias – fast path
else
  __device_stub__foo<<<…>>>();           // aliasing possible – safe path
```

Where

*   **TargetPtrs** – the pointer arguments whose indices were listed in the profile.
*   **OtherPtrs**  – all other pointer arguments.
*   The helper returns *true* when aliasing happen.

The name lookup is tolerant: if the full mangled name is not present in the profile the front-end strips the `__device_stub__` prefix and performs a suffix match so that device-side and host-side names map correctly.

### 4.3  External symbol

`check_ptr_sets` is declared but **not** defined in Clang-generated IR.  It is defined at a lib based on nvbit.

---

## 5. Hook func – `check_ptr_sets`


Signature:

```c
bool check_ptr_sets(int n_targets, void **targets,
                    int n_others,  void **others);
```

It returns *true* if *any* pointer in *targets* may alias other ptr in the two sets.


---

## 6. Full build & run recipe

```bash
# 1. collect / generate JSON profile
LD_PRELOAD=libkernel_profile.so ./gemm

# 2. compile with pass + frontend support
clang++ -fcuda-kernel-noalias \
       -mllvm -cuda-kernel-profile=profile_test.json \
       -L/path/to/libcheckkernel.so -lcheckkernel \
       -c gemm.cu -o gemm_opt

# 3. run
export LD_PRELOAD=/path/to/libcheckkernel.so
./gemm_opt
```
---

## 7. Bug修复记录

### 7.1 CGCUDARuntime.cpp中的Stub查找修复

**问题**: 在生成noalias优化的stub函数时，查找逻辑存在问题导致无法正确找到noalias版本的stub函数。

**修复位置**: `clang/lib/CodeGen/CGCUDARuntime.cpp:458`

**修复内容**:
```cpp
// 修复前：查找逻辑不正确，导致无法找到noalias版本的stub
// 修复后：正确查找noalias版本的stub函数
if (auto *NoaliasStub = CGM.getModule().getFunction(F->getName().str() + "_noalias_stub")) {
  // 使用noalias版本的stub
}
```

**影响**: 确保运行时能够正确识别和调用noalias优化版本的kernel，使得整个优化链路能够正常工作。

### 7.2 Stub与Device Kernel映射修复

**问题**: 生成的stub函数与对应的device kernel映射关系不正确，导致运行时无法正确选择优化版本。

**修复内容**: 
- 修复了stub函数命名规则，确保与device kernel名称一致
- 修复了运行时查找逻辑，确保能正确匹配优化版本
- 完善了错误处理机制，在找不到优化版本时正确回退到原始版本

**测试验证**: 通过实际测试验证了修复后的系统能够：
- ✅ 正确生成noalias优化的stub函数
- ✅ 正确生成对应的device kernel
- ✅ 运行时能正确选择kernel版本
- ✅ 在检测到别名时正确回退到原始版本

---

## 8. Limitations & future work

* Currently we only distinguish *alias* vs *no-alias* at the granularity of the whole kernel launch; finer grained specialisations (e.g. per-subset) are possible extensions.
* Consider more host-device optimization chances, like Constant Propagation from host to device, as sometimes the kernel is launched with a fixed config(args, grid and block dimensions...)
---
