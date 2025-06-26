//===- cuda_profile_runtime.c - CUDA Arguments Profiling Runtime --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the enhanced runtime library for CUDA kernel arguments 
// profiling with parameter name preservation and intelligent type detection.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <sys/time.h>
#include <unistd.h>

// 参数信息结构
typedef struct {
    char *name;
    char *type;
    int index;
} ParamInfo;

// kernel信息结构
typedef struct {
    char *kernel_name;
    int param_count;
    ParamInfo *params;
} KernelInfo;

// 全局状态
static FILE *profile_file = NULL;
static KernelInfo *kernel_registry = NULL;
static int kernel_count = 0;
static int kernel_capacity = 0;
static bool initialized = false;

// 当前kernel上下文
static char *current_kernel = NULL;
static struct timeval kernel_start_time;

// 统计信息
static int total_kernels_launched = 0;
static int total_params_profiled = 0;

// 初始化profiling系统
void init_cuda_profiling() {
    if (initialized) return;
    
    const char *filename = getenv("CUDA_PROFILE_OUTPUT");
    if (!filename) {
        filename = "cuda_profile.log";
    }
    
    profile_file = fopen(filename, "w");
    if (!profile_file) {
        fprintf(stderr, "警告：无法打开CUDA性能分析输出文件: %s，使用stderr\n", filename);
        profile_file = stderr;
    }
    
    // 写入文件头
    fprintf(profile_file, "CUDA Kernel Arguments Profile Log (Enhanced)\n");
    fprintf(profile_file, "=============================================\n");
    fprintf(profile_file, "PID: %d\n", getpid());
    
    struct timeval tv;
    gettimeofday(&tv, NULL);
    fprintf(profile_file, "启动时间: %ld.%06ld\n\n", tv.tv_sec, tv.tv_usec);
    
    // 初始化kernel注册表
    kernel_capacity = 100;
    kernel_registry = (KernelInfo*)malloc(kernel_capacity * sizeof(KernelInfo));
    if (!kernel_registry) {
        fprintf(stderr, "错误：无法分配kernel注册表内存\n");
        exit(1);
    }
    
    initialized = true;
    fprintf(profile_file, "[INFO] CUDA参数性能分析系统已初始化\n\n");
    fflush(profile_file);
}

// 清理profiling系统
void finalize_cuda_profiling() {
    if (!initialized) return;
    
    fprintf(profile_file, "\n[统计信息]\n");
    fprintf(profile_file, "总kernel启动次数: %d\n", total_kernels_launched);
    fprintf(profile_file, "总参数分析次数: %d\n", total_params_profiled);
    fprintf(profile_file, "注册的kernel数量: %d\n", kernel_count);
    
    // 清理内存
    for (int i = 0; i < kernel_count; i++) {
        free(kernel_registry[i].kernel_name);
        for (int j = 0; j < kernel_registry[i].param_count; j++) {
            free(kernel_registry[i].params[j].name);
            free(kernel_registry[i].params[j].type);
        }
        free(kernel_registry[i].params);
    }
    free(kernel_registry);
    
    if (profile_file && profile_file != stderr) {
        fclose(profile_file);
    }
    
    initialized = false;
}

// 注册kernel参数信息
void register_kernel_params(const char* kernel_name, int param_count, 
                           const char** param_names, const char** param_types) {
    init_cuda_profiling();
    
    // 检查是否已注册
    for (int i = 0; i < kernel_count; i++) {
        if (strcmp(kernel_registry[i].kernel_name, kernel_name) == 0) {
            return; // 已注册
        }
    }
    
    // 扩展数组如果需要
    if (kernel_count >= kernel_capacity) {
        kernel_capacity *= 2;
        kernel_registry = (KernelInfo*)realloc(kernel_registry, 
                                              kernel_capacity * sizeof(KernelInfo));
        if (!kernel_registry) {
            fprintf(stderr, "错误：无法重新分配kernel注册表内存\n");
            exit(1);
        }
    }
    
    // 注册新kernel
    KernelInfo *info = &kernel_registry[kernel_count];
    info->kernel_name = strdup(kernel_name);
    info->param_count = param_count;
    info->params = (ParamInfo*)malloc(param_count * sizeof(ParamInfo));
    
    for (int i = 0; i < param_count; i++) {
        info->params[i].name = strdup(param_names[i]);
        info->params[i].type = strdup(param_types[i]);
        info->params[i].index = i;
    }
    
    kernel_count++;
    
    fprintf(profile_file, "[注册] Kernel: %s, 参数数量: %d\n", kernel_name, param_count);
    for (int i = 0; i < param_count; i++) {
        fprintf(profile_file, "  参数%d: %s (%s)\n", i, param_names[i], param_types[i]);
    }
    fprintf(profile_file, "\n");
    fflush(profile_file);
}

// 查找kernel参数信息
const ParamInfo* find_param_info(const char* kernel_name, int param_index) {
    for (int i = 0; i < kernel_count; i++) {
        if (strcmp(kernel_registry[i].kernel_name, kernel_name) == 0) {
            if (param_index >= 0 && param_index < kernel_registry[i].param_count) {
                return &kernel_registry[i].params[param_index];
            }
            break;
        }
    }
    return NULL;
}

// Kernel启动profiling
void profile_kernel_launch(const char* kernel_name) {
    init_cuda_profiling();
    
    if (current_kernel) {
        free(current_kernel);
    }
    current_kernel = strdup(kernel_name);
    
    gettimeofday(&kernel_start_time, NULL);
    total_kernels_launched++;
    
    fprintf(profile_file, "\n[KERNEL #%d] %s\n", total_kernels_launched, kernel_name);
    fprintf(profile_file, "启动时间: %ld.%06ld\n", 
            kernel_start_time.tv_sec, kernel_start_time.tv_usec);
    fflush(profile_file);
}

// Grid维度profiling
void profile_grid_dim(int x, int y, int z) {
    init_cuda_profiling();
    fprintf(profile_file, "  Grid维度: (%d, %d, %d) [总线程块: %d]\n", x, y, z, x*y*z);
    fflush(profile_file);
}

// Block维度profiling
void profile_block_dim(int x, int y, int z) {
    init_cuda_profiling();
    fprintf(profile_file, "  Block维度: (%d, %d, %d) [每块线程数: %d]\n", x, y, z, x*y*z);
    fflush(profile_file);
}

// 智能标量检测
bool is_likely_scalar_value(void* ptr, int type_hint) {
    if (!ptr) return false;
    
    uintptr_t addr = (uintptr_t)ptr;
    
    // 基于type_hint的快速判断
    if (type_hint > 0 && type_hint <= 64) {
        // 整数类型，检查值是否合理
        if (type_hint <= 8) {
            int8_t val = *(int8_t*)ptr;
            return (val >= -128 && val <= 127);
        } else if (type_hint <= 16) {
            int16_t val = *(int16_t*)ptr;
            return (val >= -32768 && val <= 32767);
        } else if (type_hint <= 32) {
            int32_t val = *(int32_t*)ptr;
            return (val >= -1000000000 && val <= 1000000000);
        } else {
            int64_t val = *(int64_t*)ptr;
            return (val >= -1000000000000LL && val <= 1000000000000LL);
        }
    } else if (type_hint == 100 || type_hint == 200) {
        // 浮点类型
        if (type_hint == 100) {
            float val = *(float*)ptr;
            return (val > -1e10f && val < 1e10f && !isnan(val) && !isinf(val));
        } else {
            double val = *(double*)ptr;
            return (val > -1e15 && val < 1e15 && !isnan(val) && !isinf(val));
        }
    } else if (type_hint == 50) {
        // 半精度浮点类型
        uint16_t half_bits = *(uint16_t*)ptr;
        // 简单检查：非NaN且非Inf
        uint16_t exp = (half_bits >> 10) & 0x1F;
        return (exp != 0x1F); // 不是NaN或Inf
    }
    
    // 启发式检测：小地址值通常是标量
    if (addr < 100000) return true;
    
    // 大地址值通常是GPU指针
    if (addr > 0x100000000ULL) return false;
    
    return false; // 默认不是标量
}

// 格式化参数值为字符串
void format_param_value(void* value_ptr, int type_hint, char* buffer, size_t buffer_size) {
    if (!value_ptr) {
        snprintf(buffer, buffer_size, "NULL");
        return;
    }
    
    if (type_hint >= 1 && type_hint <= 8) {
        int8_t val = *(int8_t*)value_ptr;
        snprintf(buffer, buffer_size, "%d", val);
    } else if (type_hint > 8 && type_hint <= 16) {
        int16_t val = *(int16_t*)value_ptr;
        snprintf(buffer, buffer_size, "%d", val);
    } else if (type_hint > 16 && type_hint <= 32) {
        int32_t val = *(int32_t*)value_ptr;
        snprintf(buffer, buffer_size, "%d", val);
    } else if (type_hint > 32 && type_hint <= 64) {
        int64_t val = *(int64_t*)value_ptr;
        snprintf(buffer, buffer_size, "%ld", val);
    } else if (type_hint == 100) {
        float val = *(float*)value_ptr;
        snprintf(buffer, buffer_size, "%.6f", val);
    } else if (type_hint == 200) {
        double val = *(double*)value_ptr;
        snprintf(buffer, buffer_size, "%.6f", val);
    } else if (type_hint == 50) {
        // 半精度浮点 - 简单转换为float显示
        uint16_t half_bits = *(uint16_t*)value_ptr;
        // 这里简化处理，实际应该用proper half->float转换
        float approx_val = (float)(half_bits) / 1024.0f; // 粗略近似
        snprintf(buffer, buffer_size, "~%.3f (half)", approx_val);
    } else {
        // 未知类型，显示十六进制
        uint32_t val = *(uint32_t*)value_ptr;
        snprintf(buffer, buffer_size, "0x%08x", val);
    }
}

// 增强的标量参数profiling
void profile_scalar_arg(const char* arg_name, int arg_index, void* value_ptr, 
                       int type_info, const char* type_name) {
    init_cuda_profiling();
    total_params_profiled++;
    
    if (!value_ptr) {
        fprintf(profile_file, "  参数[%d] %s (%s): NULL\n", 
                arg_index, arg_name, type_name ? type_name : "unknown");
        fflush(profile_file);
        return;
    }
    
    // 使用注册的参数信息（如果可用）
    const ParamInfo *param_info = NULL;
    if (current_kernel) {
        param_info = find_param_info(current_kernel, arg_index);
    }
    
    const char *display_name = arg_name;
    const char *display_type = type_name ? type_name : "unknown";
    
    if (param_info) {
        display_name = param_info->name;
        display_type = param_info->type;
    }
    
    // 检查是否是合理的标量值
    if (!is_likely_scalar_value(value_ptr, type_info)) {
        fprintf(profile_file, "  参数[%d] %s (%s): <可能是指针: 0x%lx>\n", 
                arg_index, display_name, display_type, (uintptr_t)value_ptr);
        fflush(profile_file);
        return;
    }
    
    // 格式化值
    char value_str[256];
    format_param_value(value_ptr, type_info, value_str, sizeof(value_str));
    
    fprintf(profile_file, "  参数[%d] %s (%s): %s", 
            arg_index, display_name, display_type, value_str);
    
    // 添加通用的上下文信息提示
    if (param_info) {
        // 基于参数名的通用模式识别
        if (strstr(display_name, "dim") || strstr(display_name, "size") || 
            strstr(display_name, "len") || strstr(display_name, "count")) {
            fprintf(profile_file, " [维度/大小参数]");
        } else if (strstr(display_name, "alpha") || strstr(display_name, "beta") || 
                   strstr(display_name, "scale") || strstr(display_name, "factor")) {
            fprintf(profile_file, " [缩放/权重参数]");
        } else if (strstr(display_name, "offset") || strstr(display_name, "stride") ||
                   strstr(display_name, "step")) {
            fprintf(profile_file, " [偏移/步长参数]");
        } else if (strstr(display_name, "threshold") || strstr(display_name, "eps") ||
                   strstr(display_name, "tol")) {
            fprintf(profile_file, " [阈值参数]");
        }
    }
    
    fprintf(profile_file, "\n");
    fflush(profile_file);
}

// 构造函数：自动初始化
__attribute__((constructor))
static void auto_init_cuda_profiling() {
    init_cuda_profiling();
}

// 析构函数：自动清理
__attribute__((destructor))
static void auto_finalize_cuda_profiling() {
    finalize_cuda_profiling();
} 