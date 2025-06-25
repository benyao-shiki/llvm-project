//===- cuda_profile_runtime.c - CUDA Arguments Profiling Runtime --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the runtime library for CUDA kernel arguments profiling.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

// Global file handle for output
static FILE *profile_file = NULL;

// Initialize profiling (call this once at program start)
void init_cuda_profiling() {
    if (!profile_file) {
        const char *filename = getenv("CUDA_PROFILE_OUTPUT");
        if (!filename) {
            filename = "cuda_profile.log";
        }
        profile_file = fopen(filename, "w");
        if (!profile_file) {
            fprintf(stderr, "Failed to open CUDA profile output file: %s\n", filename);
            profile_file = stderr;
        }
        fprintf(profile_file, "CUDA Kernel Arguments Profile Log\n");
        fprintf(profile_file, "===================================\n\n");
        fflush(profile_file);
    }
}

// Finalize profiling (call this at program end)
void finalize_cuda_profiling() {
    if (profile_file && profile_file != stderr) {
        fclose(profile_file);
        profile_file = NULL;
    }
}

// Profile kernel launch
void profile_kernel_launch(const char* kernel_name) {
    init_cuda_profiling();
    fprintf(profile_file, "Kernel Launch: %s\n", kernel_name);
    fflush(profile_file);
}

// Profile grid dimensions
void profile_grid_dim(int x, int y, int z) {
    init_cuda_profiling();
    fprintf(profile_file, "  Grid Dimensions: (%d, %d, %d)\n", x, y, z);
    fflush(profile_file);
}

// Profile block dimensions
void profile_block_dim(int x, int y, int z) {
    init_cuda_profiling();
    fprintf(profile_file, "  Block Dimensions: (%d, %d, %d)\n", x, y, z);
    fflush(profile_file);
}

// Check if a pointer value looks like a device pointer or a scalar value
int is_likely_scalar(void* ptr) {
    uintptr_t addr = (uintptr_t)ptr;
    
    // Values that look like small integers (common scalars)
    if (addr < 100000) {
        return 1;
    }
    
    // Values that look like reasonable floats when interpreted as IEEE 754
    float float_val = *(float*)&addr;
    if (addr <= 0xFFFFFFFF && float_val > -1000000.0f && float_val < 1000000.0f && 
        float_val != 0.0f && (float_val > 0.001f || float_val < -0.001f)) {
        return 1;
    }
    
    // Large addresses are likely GPU pointers
    if (addr > 0x100000000ULL) {
        return 0;
    }
    
    return 0; // Default to not scalar
}

// Profile scalar argument with enhanced universal detection
void profile_scalar_arg(const char* arg_name, void* value_ptr, int arg_index, int type_info) {
    init_cuda_profiling();
    
    if (!value_ptr) {
        fprintf(profile_file, "  Argument %d (%s): NULL\n", arg_index, arg_name);
        fflush(profile_file);
        return;
    }
    
    fprintf(profile_file, "  Argument %d (%s): ", arg_index, arg_name);
    
    // 基于type_info的智能类型检测
    if (type_info >= 1 && type_info <= 64) {
        // 整数类型，type_info表示位宽
        if (type_info <= 8) {
            int8_t val = *(int8_t*)value_ptr;
            fprintf(profile_file, "%d (int8)", val);
        } else if (type_info <= 16) {
            int16_t val = *(int16_t*)value_ptr;
            fprintf(profile_file, "%d (int16)", val);
        } else if (type_info <= 32) {
            int32_t val = *(int32_t*)value_ptr;
            fprintf(profile_file, "%d (int32)", val);
        } else {
            int64_t val = *(int64_t*)value_ptr;
            fprintf(profile_file, "%ld (int64)", val);
        }
    } else if (type_info == 100) {
        // 单精度浮点
        float val = *(float*)value_ptr;
        fprintf(profile_file, "%.6f (float)", val);
    } else if (type_info == 200) {
        // 双精度浮点
        double val = *(double*)value_ptr;
        fprintf(profile_file, "%.6f (double)", val);
    } else {
        // 未知类型，使用启发式方法
        int32_t int_val = *(int32_t*)value_ptr;
        float float_val = *(float*)value_ptr;
        uint64_t ptr_val = *(uint64_t*)value_ptr;
        
        // 多种解释，让用户选择最合理的
        bool printed_something = false;
        
        // 检查是否像整数
        if (int_val >= -1000000 && int_val <= 1000000) {
            fprintf(profile_file, "%d", int_val);
            printed_something = true;
        }
        
        // 检查是否像浮点数
        if (float_val > -1000000.0f && float_val < 1000000.0f && 
            (float_val > 0.0001f || float_val < -0.0001f || float_val == 0.0f) &&
            !isnan(float_val) && !isinf(float_val)) {
            if (printed_something) fprintf(profile_file, " | ");
            fprintf(profile_file, "%.6f", float_val);
            printed_something = true;
        }
        
        // 检查是否像指针（但仍然输出，因为可能是大整数）
        if (ptr_val > 0x10000 && ptr_val < 0x7FFFFFFFFFFF) {
            if (printed_something) fprintf(profile_file, " | ");
            fprintf(profile_file, "0x%lx", ptr_val);
            printed_something = true;
        }
        
        if (!printed_something) {
            // 最后的fallback
            fprintf(profile_file, "0x%08x", *(uint32_t*)value_ptr);
        }
        
        fprintf(profile_file, " (auto-detected)");
    }
    
    fprintf(profile_file, "\n");
    fflush(profile_file);
}

// Constructor/destructor for automatic initialization
__attribute__((constructor))
static void auto_init_cuda_profiling() {
    init_cuda_profiling();
}

__attribute__((destructor))
static void auto_finalize_cuda_profiling() {
    finalize_cuda_profiling();
} 