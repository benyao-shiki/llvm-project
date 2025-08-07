# CudaKernelAnalysis Pass

## Overview

CudaKernelAnalysis is an LLVM analysis pass specifically designed to analyze CUDA kernel function parameters and their impact on kernel IR, providing decision-making basis for subsequent optimization passes.

## Features

### 1. CUDA Kernel Identification
- Identifies CUDA kernel functions through `ptx.kernel` attribute
- Identifies CUDA kernel functions through `PTX_Kernel` calling convention
- Automatically skips analysis for non-CUDA kernel functions

### 2. Unified Parameter Analysis
Analyzes all parameter types (scalars, pointers, and struct members) using a unified approach:
- **Scalar parameters**: Integers and floating-point numbers
- **Pointer parameters**: Pointer types for noalias analysis
- **Struct members**: Individual members of struct parameters passed by value
- **Nested structs**: Supports nested struct members with index tracking

### 3. Weight-Based Analysis
All parameters are analyzed using consistent weight calculation rules:

#### Scalar Analysis (Control Flow Importance)
- **Branch instruction usage**: Parameters directly used in `br` or `switch` instructions get weight +10
- **Comparison instruction usage**: Parameters used in comparison instructions with results used in branches get weight +5
- Higher weights indicate parameters are more important in control flow

#### Pointer Analysis (NoAlias Optimization Potential)
- **Base usage**: +1 weight for being used in the function
- **Loop usage**: +5 weight if used inside a loop
- **Memory operations**: +3 weight for load/store operations
- **Pointer arithmetic**: +2 weight for GEP operations
- **Loop-invariant usage**: +2 weight if the pointer is loop-invariant
- Higher weights indicate better candidates for `noalias` optimization

## Usage

### 1. Basic Usage

```bash
# Use in pass pipeline
./bin/opt -passes='require<cuda-kernel-analysis>' -disable-output input.ll

# Get debug output
./bin/opt -passes='require<cuda-kernel-analysis>' -debug-only=cuda-kernel-analysis -disable-output input.ll
```

### 2. Using with Other Passes

```bash
# Use before transform passes
./bin/opt -passes='require<cuda-kernel-analysis>,your-transform-pass' input.ll

# Use with multiple passes
./bin/opt -passes='require<cuda-kernel-analysis>,cuda-kernel-noalias,cuda-kernel-const' input.ll
```

### 3. Debug Options

```bash
# Enable all debug output
./bin/opt -debug -passes='require<cuda-kernel-analysis>' input.ll

# Enable only cuda-kernel-analysis debug output
./bin/opt -debug-only=cuda-kernel-analysis -passes='require<cuda-kernel-analysis>' input.ll
```

## Output Examples

### Debug Output Example

```
CudaKernelAnalysis: Analyzing CUDA kernel function 'my_cuda_kernel'
  Pointer argument 'data': weight = 11
  Scalar argument 'size': weight = 10
  Scalar argument 'threshold': weight = 5
  Struct argument 'config': analyzing 3 members
CudaKernelAnalysis: Analysis completed for function 'my_cuda_kernel'
```

### Non-CUDA Kernel Function

```
CudaKernelAnalysis: Function 'normal_function' is not a CUDA kernel, skipping analysis
```

## Analysis Results

### CudaKernelAnalysisResult Structure

```cpp
class CudaKernelAnalysisResult {
  // A map from scalar parameter info to its weight
  DenseMap<ParameterInfo, int> ScalarWeights;
  
  // A map from pointer parameter info to its weight
  DenseMap<ParameterInfo, int> PointerWeights;
  
  // Debug output flag
  bool DebugOutput;
  
  // Helper method
  SmallVector<std::pair<ParameterInfo, int>, 16> getAllParametersSorted() const;
};

struct ParameterInfo {
  std::string Name;                    // Parameter name
  const Argument *Arg;                 // Pointer to the original argument
  SmallVector<unsigned, 4> Indices;    // Member indices (length 1 for regular args, >1 for struct members)
  
  std::string getDisplayName() const;  // Get formatted display name
  bool isStructMember() const;         // Check if this is a struct member
  unsigned getArgIndex() const;        // Get the argument index
};
```

### Using in Other Passes

```cpp
// Get analysis results
auto &Result = FAM.getResult<CudaKernelAnalysis>(F);

// Access scalar parameters
for (auto [Param, Weight] : Result.ScalarWeights) {
  if (Weight > 5) {
    // Process high-weight scalar parameters
    // Param.getDisplayName() gives formatted name like "size" or "config[0]"
    // Param.getArgIndex() gives the argument index (0, 1, 2, etc.)
    // Param.isStructMember() tells if this is a struct member
  }
}

// Access pointer parameters
for (auto [Param, Weight] : Result.PointerWeights) {
  if (Weight > 8) {
    // Add noalias attributes to high-weight pointer parameters
  }
}

// Get all parameters sorted by weight
auto SortedParams = Result.getAllParametersSorted();
for (auto [Param, Weight] : SortedParams) {
  // Process parameters in weight order
}

// Access parameter information
for (auto [Param, Weight] : Result.ScalarWeights) {
  // Param.Name - parameter name
  // Param.Arg - pointer to original argument
  // Param.Indices - member indices (e.g., [0] for first arg, [1,2] for second arg's third member)
  // Param.getDisplayName() - formatted name like "data" or "config[0][1]"
  // Param.isStructMember() - true if indices length > 1
  // Param.getArgIndex() - first element of indices
}
```

## Example CUDA Kernel

```llvm
%struct.Config = type { i32, %struct.Nested, i32* }
%struct.Nested = type { float, i32 }

define void @my_cuda_kernel(i32* %data, i32 %size, %struct.Config %config) #0 {
entry:
  %0 = extractvalue %struct.Config %config, 0  ; Extract size field
  %1 = icmp sgt i32 %0, 0                      ; size field used in comparison
  br i1 %1, label %loop, label %exit           ; comparison result used in branch

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.latch ]
  %2 = extractvalue %struct.Config %config, 1  ; Extract nested struct
  %3 = extractvalue %struct.Nested %2, 0       ; Extract float field
  %4 = extractvalue %struct.Config %config, 2  ; Extract pointer field
  %val = load i32, i32* %4, align 4            ; pointer used in loop
  %cmp = icmp slt i32 %val, %0                 ; size field used in comparison
  br i1 %cmp, label %if.then, label %if.else   ; comparison result used in branch
  ; ... more code
}

attributes #0 = { "ptx.kernel"="true" }
```

Analysis Results:
- `data` (pointer): weight = 11 (base: 1, loop: 5, load: 3, loop-invariant: 2)
- `size` (scalar): weight = 10 (direct branch usage)
- `config[0]` (scalar): weight = 10 (extractvalue + comparison + branch)
- `config[1][0]` (scalar): weight = 0 (no significant usage)
- `config[2]` (pointer): weight = 11 (extractvalue + loop + load + loop-invariant)

Index System:
- `data`: indices = [0] (first argument)
- `size`: indices = [1] (second argument)
- `config[0]`: indices = [2, 0] (third argument, first member)
- `config[1][0]`: indices = [2, 1, 0] (third argument, second member, first sub-member)
- `config[2]`: indices = [2, 2] (third argument, third member)

## Implementation Details

### Key Functions

#### `analyzeScalarValue(const Value *V, Function &F, FunctionAnalysisManager &AM)`
- Generic function to analyze any scalar value (arguments or extracted values)
- Returns weight based on usage in branches and comparisons
- Higher weights for direct branch usage

#### `analyzePointerValue(const Value *V, Function &F, FunctionAnalysisManager &AM)`
- Generic function to analyze any pointer value (arguments or extracted values)
- Considers loop usage, memory operations, and loop-invariant properties
- Returns comprehensive weight for noalias decision making

#### `analyzeStructMembers(const Argument &A, unsigned ArgIndex, Function &F, FunctionAnalysisManager &AM, ...)`
- Recursively analyzes struct parameters passed by value
- Tracks `extractvalue` instructions to identify member usage
- Supports nested structs with index tracking
- Applies scalar and pointer analysis rules to individual members

### ParameterInfo Features

#### Index System
- All parameters have indices, even regular arguments
- Regular arguments: indices = [ArgIndex] (length 1)
- Struct members: indices = [ArgIndex, MemberIndex, ...] (length > 1)
- Example: `[1, 2, 3]` means second argument, third member, fourth sub-member

#### Display Names
- `getDisplayName()` provides formatted names for debugging
- Regular args: `"data"`, `"size"`
- Struct members: `"config[0]"`, `"config[1][2]"`

#### Helper Methods
- `isStructMember()`: returns true if indices length > 1
- `getArgIndex()`: returns the first element of indices (argument index)

## Compilation and Installation

### Compile LLVM

```bash
# In LLVM source directory
mkdir build && cd build
cmake -G Ninja -DLLVM_ENABLE_PROJECTS="clang;lld" -DCMAKE_BUILD_TYPE=Release ../llvm
ninja
```

### Verify Installation

```bash
# Check if pass is available
./bin/opt --print-passes | grep cuda-kernel-analysis

# Test pass functionality
./bin/opt -passes='require<cuda-kernel-analysis>' -debug-only=cuda-kernel-analysis -disable-output test.ll
```

## Notes

1. **Performance Impact**: Analysis pass only runs when needed, doesn't affect normal compilation performance
2. **Debug Output**: Use `-debug-only=cuda-kernel-analysis` for detailed debug information
3. **Dependencies**: This pass depends on `LoopAnalysis` to analyze pointer usage in loops
4. **Compatibility**: Supports all LLVM-supported CUDA kernel identification methods
5. **Struct Analysis**: Handles both named and anonymous struct types
6. **Nested Structs**: Supports arbitrary nesting depth with index tracking
7. **Index System**: All parameters have indices for consistent representation
8. **Separate Maps**: Scalar and pointer parameters are stored in separate maps for easy access

## Extensibility

The pass is designed to be extensible:
- Can add more weight calculation rules
- Can extend noalias analysis algorithms
- Can add other types of parameter analysis
- Can enhance struct member name resolution using debug information
- Can add support for array parameters
- Can extend index tracking for more complex data structures

## Test Cases

The pass includes comprehensive test cases covering:
- Loop-based kernels with pointer and scalar parameters
- Branch-based kernels with control flow dependencies
- Mixed kernels with both loop and branch characteristics
- Struct-based kernels with various member types
- Nested struct kernels with complex member access patterns
- Const-optimized versions of the same kernels

Example test kernels:
- `_Z11loop_kerneliPf`: Loop kernel with size and data parameters
- `_Z13branch_kerneliPf`: Branch kernel with flag and data parameters  
- `_Z12mixed_kerneliiPf`: Mixed kernel with size, flag, and data parameters
- `_Z15struct_kernel_config`: Struct-based kernel with Config parameter
- `_Z20nested_struct_kernel_config`: Nested struct kernel with complex member access
