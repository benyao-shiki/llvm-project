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

// Profile scalar argument with improved detection
void profile_scalar_arg(const char* arg_name, void* value_ptr, int arg_index, int unused) {
    init_cuda_profiling();
    
    if (!value_ptr) {
        return; // Skip NULL values
    }
    
    // Try to read the value safely
    // value_ptr points to the argument in the args array
    // For scalars, we need to read the actual value, not dereference as pointer
    
    // Try different scalar sizes
    // Most CUDA scalars are 4 or 8 bytes
    
    // Try as 32-bit integer first
    int32_t int_val = *(int32_t*)value_ptr;
    
    // Simple heuristic: if it's a small value, likely a scalar
    if (int_val >= -1000000 && int_val <= 1000000) {
        fprintf(profile_file, "  Scalar Argument: %s = %d\n", arg_name, int_val);
        fflush(profile_file);
        return;
    }
    
    // Try as float
    float float_val = *(float*)value_ptr;
    if (float_val > -1000000.0f && float_val < 1000000.0f && 
        (float_val > 0.001f || float_val < -0.001f || float_val == 0.0f)) {
        fprintf(profile_file, "  Scalar Argument: %s = %.6f\n", arg_name, float_val);
        fflush(profile_file);
        return;
    }
    
    // If it looks like a pointer (large address), skip it
    uintptr_t ptr_val = (uintptr_t)*(void**)value_ptr;
    if (ptr_val > 0x1000) {
        // This looks like a pointer, skip it
        return;
    }
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