# CUDA Kernel **Constant Propagation** Optimization

This document explains the design, build-time switches and run-time requirements for the *Kernel Constant Propagation* optimisation implemented in the LLVM/Clang CUDA tool-chain contained in this repository.

---

## 1. Motivation

Many CUDA kernels are launched with the same scalar parameters (like loop bounds, block dimensions, etc.) across multiple invocations. When these parameters are used in control flow (loops, branches), knowing their compile-time values enables aggressive optimizations like loop unrolling, branch elimination, and register allocation improvements.

The *Kernel Constant Propagation* pipeline provides:

1. **profilers** identify kernels and scalar arguments that commonly have the same values across launches.
2. **LLVM pass** (`CudaKernelConstPass`) clones each hot kernel and performs constant propagation on the selected parameters, enabling compiler optimizations.
3. **Clang front-end** (`CGCUDARuntime`) inserts a run-time check that compares actual parameter values with expected constants and chooses the appropriate kernel version.
4. A hook func (`check_const`) provides the constant value check at run time.

Example optimizations enabled:
- Loop with constant bound `for(i=0; i<size; i++)` where `size=1` → loop can be unrolled or eliminated
- Branch with constant condition `if(flag)` where `flag=1` → dead code elimination

The result is a *fully automatic*, profile-guided optimisation that yields performance improvements while being 100% safe – when parameter values don't match expected constants, we simply fall back to the unoptimised kernel.

---

## 2. JSON profile format

The optimiser consumes a single JSON file (e.g. `profile_test.json`). A reduced example:

```json
{
  "hot_kernels": [
    {
      "name": "_Z11gemm_kerneliiiffPfS_S_",
      "demangled_name": "gemm_kernel(int, int, int, float, float, float*, float*, float*)",
      "launches": 1,
      "common_scalars": [
        {
          "arg": 0,
          "value": "512",
          "ratio": 1.0
        },
        {
          "arg": 1,
          "value": "1",
          "ratio": 0.95
        }
      ],
      "common_dims": [
        {
          "dim": "gridDim.x",
          "value": 16,
          "ratio": 1.0
        },
        {
          "dim": "blockDim.x", 
          "value": 256,
          "ratio": 0.9
        }
      ]
    }
  ]
}
```

Keys:

* `hot_kernels[]` – kernels that should be specialized.
* `name` – *device* side mangled name (same as shown in PTX / `nvcc --ptxas-options=-v`).
* `common_scalars[]` – list of scalar arguments (zero-based index) that frequently have the same values.
  * `arg` – parameter index (0-based)
  * `value` – the common value as string
  * `ratio` – frequency ratio (0.0 to 1.0)
* `common_dims[]` – list of common launch dimensions (future extension).

If either the kernel is absent from the array or its `common_scalars` list is empty, that kernel is left untouched.

---

## 3. LLVM Pass – `CudaKernelConstPass`

Location: `llvm/lib/Transforms/CudaKernelConst/`.

Activation:

```bash
clang++ -mllvm -cuda-kernel-const-profile=/path/to/profile_test.json …
```

Behaviour:

1. Scans every function with calling convention `ptx_kernel` (or with `nvvm.annotations` == "kernel").
2. Looks up the function name in the parsed profile map.
3. Checks if any common scalar parameter is used in control flow (loops, branches).
4. If optimization is beneficial:
   * Clones the function with name `<orig>_const`.
   * Replaces the specified parameters with constant values in the clone **only**.
   * Copies the original `nvvm.annotations` entry so that the clone is also visible to the GPU driver.

---

## 4. Clang Front-End Support (`CGCUDARuntime.cpp`)

Compile-time switch:

```
–fcuda-kernel-const   # sets CodeGenOpts.CudaKernelConst
```

### 4.1 Profile loading

`loadCudaKernelProfile()` loads the same JSON file and populates both noalias and constant propagation profile maps. The path is resolved in the following order:

1. Environment variable `CUDA_KERNEL_PROFILE`.
2. Last occurrence of the backend option `-cuda-kernel-const-profile=<file>`.

### 4.2 Launch-site transformation

At every host-side kernel call, when constant propagation is enabled, Clang emits:

```
if (check_const(NumValues, ExpectedValues, ActualValues))
  __device_stub__foo_const<<<…>>>();   // values match – optimized path
else
  __device_stub__foo<<<…>>>();         // values differ – safe path
```

Where:

* **ExpectedValues** – the expected constant values from the profile.
* **ActualValues** – the actual parameter values at runtime.
* The helper returns *true* when all values match the expected constants.

### 4.3 External symbol

`check_const` is declared but **not** defined in Clang-generated IR. It should be defined in an external library.

---

## 5. Hook func – `check_const`

Signature:

```c
bool check_const(int n_values, int64_t *expected, int64_t *actual);
```

It returns *true* if all values in *actual* match the corresponding values in *expected*.

---

## 6. Full build & run recipe

```bash
# 1. collect / generate JSON profile
LD_PRELOAD=libkernel_profile.so ./gemm

# 2. compile with pass + frontend support
clang++ -fcuda-kernel-const \
       -mllvm -cuda-kernel-const-profile=profile_test.json \
       -L/path/to/libcheckkernel.so -lcheckkernel \
       -c gemm.cu -o gemm_opt

# 3. run
export LD_PRELOAD=/path/to/libcheckkernel.so
./gemm_opt
```

---

## 7. Optimization Examples

### 7.1 Loop Optimization

```cuda
__global__ void kernel(int size, float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = idx; i < size; i += gridDim.x * blockDim.x) {
        data[i] = data[i] * 2.0f;
    }
}
```

If profile shows `size=1` with high frequency, the constant propagation creates:

```cuda
__global__ void kernel_const(int size, float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = idx; i < 1; i += gridDim.x * blockDim.x) {  // size=1
        data[i] = data[i] * 2.0f;
    }
}
```

Compiler can then optimize this to eliminate the loop overhead.

### 7.2 Branch Optimization

```cuda
__global__ void kernel(int flag, float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (flag) {
        data[idx] = data[idx] * 2.0f;
    } else {
        data[idx] = data[idx] + 1.0f;
    }
}
```

If profile shows `flag=1` with high frequency, constant propagation enables dead code elimination.

### 7.3 Real-world LLVM IR Optimization Results

Using the test kernels in `build/yb_test/`, here are the actual LLVM IR optimizations achieved:

#### **loop_kernel** - Loop Bound Optimization
**Original LLVM IR:**
```llvm
%cmp14 = icmp slt i32 %add, %size
```

**Optimized LLVM IR (_const version):**
```llvm
%cmp1 = icmp slt i32 %add, 1
```

**Effect:** Loop condition changed from variable comparison to constant comparison, enabling further loop optimizations.

#### **branch_kernel** - Complete Branch Elimination
**Original LLVM IR (8 instructions):**
```llvm
%tobool.not = icmp eq i32 %flag, 0
%idxprom6 = sext i32 %add to i64
%arrayidx7 = getelementptr inbounds float, ptr %data, i64 %idxprom6
%3 = load float, ptr %arrayidx7, align 4
%add8 = fadd contract float %3, 1.000000e+00
%mul3 = fmul contract float %3, 2.000000e+00
%add8.sink = select i1 %tobool.not, float %add8, float %mul3
store float %add8.sink, ptr %arrayidx7, align 4
```

**Optimized LLVM IR (5 instructions):**
```llvm
%idxprom = sext i32 %add to i64
%arrayidx = getelementptr inbounds float, ptr %data, i64 %idxprom
%3 = load float, ptr %arrayidx, align 4
%mul3 = fmul contract float %3, 2.000000e+00
store float %mul3, ptr %arrayidx, align 4
```

**Effect:** Complete elimination of branch logic (37.5% instruction reduction). Since `flag=1` is constant, the compiler directly generates the multiplication path, removing conditional logic entirely.

#### **mixed_kernel** - Combined Loop and Branch Optimization
**Original LLVM IR:**
```llvm
%tobool.not = icmp eq i32 %flag, 0
%cmp10 = icmp slt i32 %add, %size
br i1 %tobool.not, label %if.else, label %for.cond.preheader
```

**Optimized LLVM IR:**
```llvm
%cmp1 = icmp slt i32 %add, 1
br i1 %cmp1, label %for.body.lr.ph, label %if.end17
```

**Effect:** Both branch elimination (direct to `flag=1` path) and loop bound optimization (`size` → `1`), removing the `if.else` branch completely.

These optimizations are particularly beneficial for GPU architectures where branch divergence penalties are high.

---

## 8. Limitations & future work

* Currently focuses on scalar parameters; dimension constants are parsed but not yet fully implemented.
* Control flow analysis is basic; more sophisticated analysis could identify additional optimization opportunities.
* Consider integration with other profile-guided optimizations for compound effects.

--- 