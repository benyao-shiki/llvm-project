//===- CudaArgsProfileRuntime.c - Runtime for CUDA Args Profiling -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runtime library for CUDA kernel arguments profiling
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAX_KERNELS 100
#define MAX_PARAMS 20

// Type enum values (must match the ones in the pass)
typedef enum {
  SCALAR_INT8 = 0, SCALAR_INT16 = 1, SCALAR_INT32 = 2, SCALAR_INT64 = 3,
  SCALAR_FLOAT = 4, SCALAR_DOUBLE = 5,
  POINTER = 6,
  STRUCT = 7
} ParamType;

typedef struct {
  int index;
  int type;
  size_t size;
  char value[256];
  char struct_name[256];  // For struct types
  int member_count;       // Number of members for struct
  int is_struct_member;   // 1 if this is a struct member, 0 otherwise
  int parent_index;       // Index of parent struct (for members)
} ParamValue;

typedef struct {
  int id;
  char name[256];
  int grid[3];
  int block[3];
  int param_count;
  ParamValue params[MAX_PARAMS];
} KernelLaunch;

typedef struct {
  char name[256];
  int launches;
} KernelStats;

// Global state
static KernelLaunch kernel_launches[MAX_KERNELS];
static KernelStats kernel_stats[MAX_KERNELS];
static int kernel_count = 0;
static int stats_count = 0;
static KernelLaunch current_kernel;
static int profiling_enabled = 1;

// Utility functions
static void format_hex(void* ptr, char* buffer) {
  sprintf(buffer, "0x%lx", (unsigned long)ptr);
}

static void format_scalar_value(int type, void* value_ptr, size_t size, char* buffer) {
  // Handle NULL pointer
  if (value_ptr == NULL) {
    sprintf(buffer, "null");
    return;
  }
  
  switch (type) {
    case SCALAR_INT8: {
      int8_t val = *(int8_t*)value_ptr;
      sprintf(buffer, "%d", (int)val);
      break;
    }
    case SCALAR_INT16: {
      int16_t val = *(int16_t*)value_ptr;
      sprintf(buffer, "%d", val);
      break;
    }
    case SCALAR_INT32: {
      int32_t val = *(int32_t*)value_ptr;
      sprintf(buffer, "%d", val);
      break;
    }
    case SCALAR_INT64: {
      int64_t val = *(int64_t*)value_ptr;
      sprintf(buffer, "%lld", (long long)val);
      break;
    }
    case SCALAR_FLOAT: {
      float val = *(float*)value_ptr;
      uint32_t hex_val = *(uint32_t*)&val;
      sprintf(buffer, "%u", hex_val);
      break;
    }
    case SCALAR_DOUBLE: {
      double val = *(double*)value_ptr;
      uint64_t hex_val = *(uint64_t*)&val;
      sprintf(buffer, "%llu", (unsigned long long)hex_val);
      break;
    }
    default:
      sprintf(buffer, "unknown");
      break;
  }
}

static void dims_to_string(int dims[3], char* buffer) {
  sprintf(buffer, "%dx%dx%d", dims[0], dims[1], dims[2]);
}

static KernelStats* find_or_create_stats(const char* name) {
  // Find existing stats
  for (int i = 0; i < stats_count; i++) {
    if (strcmp(kernel_stats[i].name, name) == 0) {
      return &kernel_stats[i];
    }
  }
  
  // Create new stats
  if (stats_count < MAX_KERNELS) {
    strcpy(kernel_stats[stats_count].name, name);
    kernel_stats[stats_count].launches = 0;
    return &kernel_stats[stats_count++];
  }
  
  return NULL;
}

// Runtime API functions
void __cuda_profile_kernel_start(const char* kernel_info, int kernel_id, 
                                 int gridx, int gridy, int gridz,
                                 int blockx, int blocky, int blockz) {
  if (!profiling_enabled) return;
  
  // Parse kernel info string: "kernel_name|param_count|..."
  char* info_copy = strdup(kernel_info);
  char* token = strtok(info_copy, "|");
  if (!token) {
    free(info_copy);
    return;
  }
  
  // Initialize current kernel
  current_kernel.id = kernel_id;
  strcpy(current_kernel.name, token);
  current_kernel.grid[0] = gridx;
  current_kernel.grid[1] = gridy;
  current_kernel.grid[2] = gridz;
  current_kernel.block[0] = blockx;
  current_kernel.block[1] = blocky;
  current_kernel.block[2] = blockz;
  current_kernel.param_count = 0;
  
  // Update stats
  KernelStats* stats = find_or_create_stats(current_kernel.name);
  if (stats) {
    stats->launches++;
  }
  
  free(info_copy);
}

void __cuda_profile_kernel_end(void) {
  if (!profiling_enabled) return;
  
  // Add current kernel to the list
  if (kernel_count < MAX_KERNELS) {
    kernel_launches[kernel_count++] = current_kernel;
  }
}

void __cuda_profile_scalar(int index, int type, void* value_ptr, size_t size) {
  if (!profiling_enabled) {
    return;
  }
  
  // Check if this is a struct member (index >= 100)
  int is_member = (index >= 100);
  int parent_index = -1;
  int member_index = index;
  
  if (is_member) {
    parent_index = index / 100;
    member_index = index % 100;
    
    // Find next available slot for the member
    for (int i = 0; i < MAX_PARAMS; i++) {
      if (current_kernel.params[i].index == 0 && current_kernel.params[i].type == 0) {
        member_index = i;
        break;
      }
    }
    
    if (member_index >= MAX_PARAMS) {
      return;
    }
    
    // Update parent struct member count
    if (parent_index < MAX_PARAMS) {
      current_kernel.params[parent_index].member_count++;
    }
  } else {
    if (index >= MAX_PARAMS) {
      return;
    }
  }
  
  ParamValue* param = &current_kernel.params[member_index];
  param->index = index;  // Keep original index for reference
  param->type = type;
  param->size = size;
  param->is_struct_member = is_member;
  param->parent_index = parent_index;
  param->struct_name[0] = '\0';  // Clear struct name for scalars
  
  format_scalar_value(type, value_ptr, size, param->value);
  
  if (!is_member && index >= current_kernel.param_count) {
    current_kernel.param_count = index + 1;
  }
}

void __cuda_profile_struct_value(int index, const char* struct_name, void* value_ptr, size_t size, const char* member_info) {
  if (!profiling_enabled) return;
  
  if (index >= MAX_PARAMS) return;
  
  ParamValue* param = &current_kernel.params[index];
  param->index = index;
  param->type = STRUCT;
  param->size = size;
  strcpy(param->struct_name, struct_name);
  param->is_struct_member = 0;
  param->parent_index = -1;
  
  // Parse member info string: "member_count:type1:offset1:size1:type2:offset2:size2:..."
  char* info_copy = strdup(member_info);
  char* token = strtok(info_copy, ":");
  
  int member_count = 0;
  if (token) {
    member_count = atoi(token);
    param->member_count = member_count;
  }
  
  // Store the raw struct value as hex for reference
  if (size == 8) {
    uint64_t val = *(uint64_t*)value_ptr;
    sprintf(param->value, "%llu", (unsigned long long)val);
  } else if (size == 4) {
    uint32_t val = *(uint32_t*)value_ptr;
    sprintf(param->value, "%u", val);
  } else {
    // For other sizes, format as hex
    sprintf(param->value, "0x");
    for (size_t i = 0; i < size && i < 32; i++) {
      sprintf(param->value + 2 + i*2, "%02x", ((uint8_t*)value_ptr)[i]);
    }
  }
  
  // Parse and store struct members
  for (int i = 0; i < member_count && i < MAX_PARAMS - index - 1; i++) {
    int member_type = -1, member_offset = -1, member_size = -1;
    
    token = strtok(NULL, ":");
    if (token) member_type = atoi(token);
    
    token = strtok(NULL, ":");
    if (token) member_offset = atoi(token);
    
    token = strtok(NULL, ":");
    if (token) member_size = atoi(token);
    
    if (member_type >= 0 && member_offset >= 0 && member_size > 0) {
      // Find next available slot for member
      int member_slot = -1;
      for (int j = index + 1; j < MAX_PARAMS; j++) {
        if (current_kernel.params[j].index == 0 && current_kernel.params[j].type == 0) {
          member_slot = j;
          break;
        }
      }
      
      if (member_slot >= 0) {
        ParamValue* member_param = &current_kernel.params[member_slot];
        member_param->index = index * 100 + i; // Special index for struct members
        member_param->type = member_type;
        member_param->size = member_size;
        member_param->is_struct_member = 1;
        member_param->parent_index = index;
        member_param->struct_name[0] = '\0';
        
        // Extract member value from struct
        void* member_ptr = ((uint8_t*)value_ptr) + member_offset;
        format_scalar_value(member_type, member_ptr, member_size, member_param->value);
      }
    }
  }
  
  free(info_copy);
  
  if (index >= current_kernel.param_count) {
    current_kernel.param_count = index + 1;
  }
}

void __cuda_profile_struct_start(int index, const char* struct_name, size_t size) {
  if (!profiling_enabled) return;
  
  if (index >= MAX_PARAMS) return;
  
  ParamValue* param = &current_kernel.params[index];
  param->index = index;
  param->type = STRUCT;
  param->size = size;
  strcpy(param->struct_name, struct_name);
  sprintf(param->value, "struct_%s", struct_name);
  param->member_count = 0;
  param->is_struct_member = 0;
  param->parent_index = -1;
  
  if (index >= current_kernel.param_count) {
    current_kernel.param_count = index + 1;
  }
}

void __cuda_profile_pointer(int index, void* ptr_value) {
  if (!profiling_enabled) return;
  
  if (index >= MAX_PARAMS) return;
  
  ParamValue* param = &current_kernel.params[index];
  param->index = index;
  param->type = POINTER;
  param->size = sizeof(void*);
  format_hex(ptr_value, param->value);
  
  if (index >= current_kernel.param_count) {
    current_kernel.param_count = index + 1;
  }
}

void __cuda_profile_finalize(void) {
  if (!profiling_enabled) return;
  
  // Generate JSON output
  FILE* fp = fopen("cuda_profile.json", "w");
  if (!fp) {
    fprintf(stderr, "Error: Cannot open cuda_profile.json for writing\n");
    return;
  }
  
  fprintf(fp, "{\n");
  fprintf(fp, "  \"total_kernels\": %d,\n", stats_count);
  
  // Hot kernels section
  fprintf(fp, "  \"hot_kernels\": [\n");
  for (int i = 0; i < stats_count; i++) {
    if (i > 0) fprintf(fp, ",\n");
    
    fprintf(fp, "    {\n");
    fprintf(fp, "      \"name\": \"%s\",\n", kernel_stats[i].name);
    fprintf(fp, "      \"launches\": %d,\n", kernel_stats[i].launches);
    fprintf(fp, "      \"common_scalars\": [],\n");
    fprintf(fp, "      \"common_dims\": [],\n");
    fprintf(fp, "      \"noalias_pointers\": []\n");
    fprintf(fp, "    }");
  }
  fprintf(fp, "\n  ],\n");
  
  // Kernels section
  fprintf(fp, "  \"kernels\": [\n");
  for (int i = 0; i < kernel_count; i++) {
    if (i > 0) fprintf(fp, ",\n");
    
    KernelLaunch* launch = &kernel_launches[i];
    fprintf(fp, "    {\n");
    fprintf(fp, "      \"id\": %d,\n", launch->id);
    fprintf(fp, "      \"name\": \"%s\",\n", launch->name);
    fprintf(fp, "      \"grid\": [%d,%d,%d],\n", 
            launch->grid[0], launch->grid[1], launch->grid[2]);
    fprintf(fp, "      \"block\": [%d,%d,%d],\n", 
            launch->block[0], launch->block[1], launch->block[2]);
    fprintf(fp, "      \"shmem_static\": 0,\n");
    fprintf(fp, "      \"shmem_dynamic\": 0,\n");
    fprintf(fp, "      \"registers\": 32,\n");
    
    // Parameters
    fprintf(fp, "      \"params\": [");
    int first_param = 1;
    for (int j = 0; j < launch->param_count; j++) {
      ParamValue* param = &launch->params[j];
      
      // Skip struct members in main parameter list (they will be included in struct)
      if (param->is_struct_member) continue;
      
      if (!first_param) fprintf(fp, ",");
      first_param = 0;
      
      if (param->type == STRUCT) {
        // Output struct parameter with raw value and parsed members
        fprintf(fp, "{\"type\": \"struct\", \"name\": \"%s\", \"size\": %zu, \"raw_value\": \"%s\", \"members\": [", 
                param->struct_name, param->size, param->value);
        
        // Find and output struct members
        int first_member = 1;
        for (int k = 0; k < MAX_PARAMS; k++) {
          ParamValue* member = &launch->params[k];
          if (member->is_struct_member && member->parent_index == j) {
            if (!first_member) fprintf(fp, ",");
            first_member = 0;
            fprintf(fp, "{\"size\": %zu, \"value\": \"%s\"}", 
                    member->size, member->value);
          }
        }
        fprintf(fp, "]}");
      } else {
        // Output regular parameter
        fprintf(fp, "{\"size\": %zu, \"value\": \"%s\"}", 
                param->size, param->value);
      }
    }
    fprintf(fp, "]\n");
    
    fprintf(fp, "    }");
  }
  fprintf(fp, "\n  ]\n");
  
  fprintf(fp, "}\n");
  fclose(fp);
  
  printf("CUDA arguments profiling results written to cuda_profile.json\n");
} 