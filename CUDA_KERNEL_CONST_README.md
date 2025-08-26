# CUDA Kernel **Constant Propagation** Optimization

This document explains the design, build-time switches and run-time requirements for the Kernel Constant Propagation optimisation implemented in the LLVM/Clang CUDA tool-chain contained in this repository.

---

## 1. Motivation

Many CUDA kernels are launched with the same scalar parameters (like loop bounds, block dimensions, etc.) across multiple invocations. When these parameters are used in control flow (loops, branches), knowing their compile-time values enables aggressive optimizations like loop unrolling, branch elimination, and register allocation improvements.

The Kernel Constant Propagation pipeline provides:

1. profilers identify kernels and scalar arguments (including struct members) that commonly have the same values across launches.
2. LLVM pass (`CudaKernelConstPass`) clones each hot kernel and performs constant propagation on the selected parameters, enabling compiler optimizations.
3. Clang front-end (`CGCUDARuntime`) inserts a run-time check that compares actual parameter values with expected constants and chooses the appropriate kernel version.
4. A hook func (`check_const`) provides the constant value check at run time.

Example optimizations enabled:
- Loop with constant bound `for(i=0; i<size; i++)` where `size=1` → loop can be unrolled or eliminated
- Branch with constant condition `if(flag)` where `flag=1` → dead code elimination
- Struct-by-value member used in control-flow, e.g. `s.value` → constant-fold the member

---

## 2. JSON profile format

The optimiser consumes a single JSON file (e.g. `profile.json`). The pass no longer depends on the deprecated `hot_kernels` summary. Instead, it computes the joint frequency of parameter combinations from detailed per-launch records in `kernels[]` and selects the best subset (union-mode).

Reduced example showing per-launch params (including struct members):

```json
{
  "kernels": [
    {
      "name": "__device_stub__fill_runtime",
      "block": [1,1,1],
      "grid": [1,1,1],
      "params": [
        { "index": 0, "name": "x",   "type": "int*",    "value": "(nil)" },
        { "index": 1, "name": "s",   "type": "MyStruct", "value": [
            { "index": 0, "name": "a",     "type": "int", "value": 0 },
            { "index": 1, "name": "value", "type": "int", "value": 1 }
        ]},
        { "index": 2, "name": "size", "type": "int",     "value": 1 }
      ]
    }
  ]
}
```

Notes:
- Each launch is flattened into a map from an index-path string to value. For scalars the path is flat index (e.g. `"2"`). For struct/nested members the path is `arg.member...` indices joined by dots (e.g. `"1.1"` for argument 1 member 1).
- Kernel names are normalised to match device functions (e.g. stripping `__device_stub__`).

---

## 3. LLVM Pass – `CudaKernelConstPass`

Location: `llvm/lib/Transforms/CudaKernelConst/`.

### Activation

```bash
# Enable in clang via backend options
clang++ -fcuda-kernel-const \
        -mllvm -cuda-kernel-const-profile=/path/to/profile.json …
```

### Behaviour

1. Scans every function with calling convention `ptx_kernel` (or with `nvvm.annotations` == "kernel").
2. Looks up the function name in the parsed profile map.
3. Uses `CudaKernelAnalysis` to compute weights for parameters (and nested struct members). Only parameters with weight > 0 are considered as candidates.
4. Enumerates all non-empty subsets of candidates. For each subset, computes a joint frequency (union-mode) from all `kernels[]` launches. Scores the subset as `score = Σ (weight_i * ratio_joint)` and prefers larger subsets on ties. Chooses the best subset.
5. Emits selection metadata for the best-scoring subset, and clones the function as `<orig>_const`, replacing those parameters with constants in the clone only.
6. Copies the original `nvvm.annotations` entry so that the clone remains a kernel.

### Selection metadata schema

The pass writes a module-level named metadata `!cuda.const.selected` for host-side use. Each selected parameter produces one record:

```llvm
!cuda.const.selected = !{ !REC0, !REC1, ... }
!REC0 = !{ ptr @kernel, !"scalar", !"<name>", !"<value>", !INDICES }
!INDICES = !{ i32 <arg>, [i32 <member> ...] }
```

- `@kernel`: the device function being specialized
- `"scalar"`: kind tag (future-proof)
- `<name>`: stable parameter name from profile if available, else empty
- `<value>`: the chosen constant value as string
- `INDICES`: full index vector – flat argument index followed by zero or more struct member indices (e.g. `{ i32 1, i32 1 }` means `arg1.member1`)

### Implementation notes (pass)

- Profile parsing: Uses `kernels[]` records only; discards legacy `hot_kernels`.
- Candidate enumeration: builds a vector of `(ArgIndex, IndicesPath, Weight)` from `CudaKernelAnalysis` results (weights > 0).
- Joint-frequency (union-mode): For each subset, builds a key by concatenating the values observed at the subset’s paths for a launch; the mode (most frequent) combination and its ratio drive the score.
- Constant creation: parses integers/floats without exceptions (toolchain builds with `-fno-exceptions`). For integers, accepts decimal or rounds floating strings via `strtod`.
- Replacement rules in clone:
  - Top-level scalars: replace all uses of the `Argument` with a `Constant`.
  - Struct member scalars: replace matching `extractvalue` uses, and also match constant-index `getelementptr` + `load` (common lowering for byval aggregates) and replace their uses with a `Constant`.
- Debug: `-mllvm -cuda-kernel-const-debug` prints subset scores and final selection, including full index paths and chosen values.

---

## 4. Clang Front-End Support (`CGCUDARuntime.cpp`)

### Compile-time switch

```
-fcuda-kernel-const   # sets CodeGenOpts.CudaKernelConst
```

### 4.1 Profile loading

`loadCudaKernelProfile()` loads the same JSON file and fills a map of selected constants. It supports two shapes of `cuda_const_selected` (object keyed by kernel, or array with `name` entries). For each item, it now preserves the full `indices` path along with the string value.

Resolution order for profile path:
1. Environment variable `CUDA_KERNEL_PROFILE`
2. Backend option `-cuda-kernel-const-profile=<file>` (last occurrence)

### 4.2 Launch-site transformation

At every host-side kernel call, when constant propagation is enabled, Clang emits:

```
if (check_const(NumValues, ExpectedValues, ActualValues))
  __device_stub__foo_const<<<…>>>();   // values match – optimized path
else
  __device_stub__foo<<<…>>>();         // values differ – safe path
```

Where:
- ExpectedValues – taken from the profile selections
- ActualValues – computed at runtime from the call-site arguments

For struct-by-value members, the front-end uses the preserved full `indices`:
- Top-level scalar: `EmitAnyExpr(arg).getScalarVal()`
- Struct member path: start from `EmitLValue(arg)`, walk fields/arrays to the member `Address`, then `EmitLoadOfScalar(Address, …)` to get the scalar. This avoids calling `getScalarVal()` on aggregate rvalues and prevents the previous assertion.

`check_const` is declared but not defined in Clang-generated IR; provide it via an external library.

---

## 5. Hook func – `check_const`

Signature:

```c
bool check_const(int n_values, int64_t *expected, int64_t *actual);
```

It returns true if all values in `actual` match the corresponding values in `expected`.

---

## 6. Test case and expected effects

We use `build/yb_test/test_const.cu`:

```cuda
struct MyStruct { int a; int value; };
extern "C" __global__ void fill_runtime(int *x, MyStruct s, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = idx; i < size; i += gridDim.x * blockDim.x) {
    x[i] *= s.value;
  }
}
```

- Profile launches record `s.value = 1` and `size = 1` (paths `1.1` and `2`).
- Analysis weights mark `s.value` and `size` as candidates.
- Const pass selects the joint best subset `{ (1.1)=1, (2)=1 }` and emits `!cuda.const.selected` with indices `[1,1]` and `[2]`.
- The pass clones `fill_runtime` to `fill_runtime_const` and in the clone:
  - Replaces uses of `size` with constant `1`.
  - Replaces `extractvalue s, 1` (or byval-lowered `gep+load` of `s.value`) with constant `1`.
- Front-end emits a run-time check at launch site to dispatch to `__device_stub__fill_runtime_const` when values match.

### Sample debug output

```
[CudaKernelConst] kernel=fill_runtime subset_mask=0x3 ratio=1.000000e+00 score=1.300000e+01 combo_count=1/1 combo={arg1(path=1.1)=1.000000, arg2(path=2)=1.000000}
[CudaKernelConst] kernel=fill_runtime best_mask=0x3 best_score=1.300000e+01 selected={arg1(path=1.1)=1.000000 (single_ratio=1.000000e+00), arg2(path=2)=1.000000 (single_ratio=1.000000e+00)}
```

---

## 7. Full build & run recipe

```bash
# 1) build clang/llvm with this repo
# … standard cmake+ninja steps …

# 2) compile the test with the pass + frontend support
env CUDA_KERNEL_PROFILE=/path/to/profile.json \
clang++ test_const.cu --cuda-gpu-arch=sm_80 \
        -fcuda-kernel-const \
        -mllvm -cuda-kernel-const-profile=/path/to/profile.json \
        -mllvm -cuda-kernel-const-debug \
        -L/path/to -lcheckkernel -lcudart -o test_const

# 3) run (ensure check_const is available)
LD_PRELOAD=/path/to/libcheckkernel.so ./test_const
```

---

## 8. Limitations & future work

- Dimension constants for grid/block are parsed but not yet propagated.
- The analysis heuristics are intentionally simple; can be enhanced.
- More IR shapes for struct-member loads (e.g., PHI-propagated) can be recognized for replacement. 