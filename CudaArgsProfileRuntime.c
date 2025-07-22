//===- CudaArgsProfileRuntime.c - Runtime for CUDA Args Profiling -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runtime library for CUDA kernel arguments profiling.
// This version receives detailed type information for each argument just
// before the kernel launch.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAX_KERNELS 128
#define MAX_PARAMS 64
#define MAX_MEMBERS 64

// Forward declaration for nested structs
struct ParamInfoV2;

// Holds information about a single struct member
typedef struct MemberInfoV2 {
    char name[64];
    char type_name[128];
    char type_kind; // 's' (scalar), 'p' (pointer), 'r' (record/struct)
    int size;
    int offset;
    int num_nested_members;
    struct ParamInfoV2* nested_members; // Points to an array of ParamInfoV2 for nested structs
} MemberInfoV2;

// Holds information about a single kernel parameter
typedef struct ParamInfoV2 {
    char name[64];
    char type_name[128];
    char type_kind;
    int size;
    int num_members;
    MemberInfoV2 members[MAX_MEMBERS];
} ParamInfoV2;

// Holds the arguments for a single kernel launch
typedef struct {
    int id;
    char name[256];
    int grid[3];
    int block[3];
    int num_params;
    void* arg_values[MAX_PARAMS]; // Store pointers to COPIED argument values
    char* arg_signatures[MAX_PARAMS]; // Store COPIED signature strings
} KernelLaunch;

// Global state
static KernelLaunch kernel_launches[MAX_KERNELS];
static int launch_count = 0;

static KernelLaunch current_kernel;
static int profiling_enabled = 1;


// --- Signature Parsing Helpers ---

static char* parse_members(char* sig_str, MemberInfoV2* members, int* num_members);

// Parses type info: type_name:kind:size[:{members}]
static char* parse_type_info(char* sig_str, ParamInfoV2* param) {
    char* next_token = strchr(sig_str, ':');
    if (!next_token) return NULL;
    *next_token = '\0';
    strncpy(param->type_name, sig_str, sizeof(param->type_name) - 1);
    param->type_name[sizeof(param->type_name) - 1] = '\0'; // Ensure null termination
    sig_str = next_token + 1;

    next_token = strchr(sig_str, ':');
    if (!next_token) return NULL;
    *next_token = '\0';
    param->type_kind = sig_str[0];
    sig_str = next_token + 1;

    char* end_of_size_num = sig_str;
    while (*end_of_size_num >= '0' && *end_of_size_num <= '9') {
        end_of_size_num++;
    }
    char temp_char_after_size = *end_of_size_num; // Save the char after the size number
    *end_of_size_num = '\0'; // Null-terminate the size string
    param->size = atoi(sig_str);
    *end_of_size_num = temp_char_after_size; // Restore the char

    sig_str = end_of_size_num; // Advance sig_str to the character after the size number

    if (param->type_kind == 'r') {
        if (*sig_str == ':') { // Expect ':' before '{' for nested struct type name
            sig_str++; // Move past ':'
        }
        if (*sig_str == '{') {
            sig_str = parse_members(sig_str + 1, param->members, &param->num_members);
        }
    }
    return sig_str;
}

// Parses a member list: member1,member2,...}
static char* parse_members(char* sig_str, MemberInfoV2* members, int* num_members) {
    *num_members = 0;
    while (*sig_str != '}' && *sig_str != '\0') {
        if (*num_members >= MAX_MEMBERS) break;
        MemberInfoV2* member = &members[(*num_members)++];

        // 解析成员名
        char* next_token = strchr(sig_str, ':');
        if (!next_token) break;
        *next_token = '\0';
        strncpy(member->name, sig_str, sizeof(member->name) - 1);
        member->name[sizeof(member->name) - 1] = '\0';
        sig_str = next_token + 1;

        // 解析类型名
        next_token = strchr(sig_str, ':');
        if (!next_token) break;
        *next_token = '\0';
        strncpy(member->type_name, sig_str, sizeof(member->type_name) - 1);
        member->type_name[sizeof(member->type_name) - 1] = '\0';
        sig_str = next_token + 1;

        // 解析kind
        member->type_kind = *sig_str;
        sig_str += 2; // 跳过 kind 和下一个冒号

        // 解析size
        int size = 0;
        while (*sig_str >= '0' && *sig_str <= '9') {
            size = size * 10 + (*sig_str - '0');
            sig_str++;
        }
        member->size = size;
        if (*sig_str == ':') sig_str++;

        // 递归解析struct成员
        if (member->type_kind == 'r' && *sig_str == '{') {
            // 递归到一个临时ParamInfoV2，然后拷贝其members到nested_members
            ParamInfoV2 temp_param = {0};
            sig_str = parse_members(sig_str + 1, temp_param.members, &temp_param.num_members);
            member->num_nested_members = temp_param.num_members;
            if (temp_param.num_members > 0) {
                member->nested_members = (struct ParamInfoV2*)malloc(sizeof(ParamInfoV2));
                if (member->nested_members) {
                    memcpy(member->nested_members, &temp_param, sizeof(ParamInfoV2));
                }
            } else {
                member->nested_members = NULL;
            }
        } else {
            member->num_nested_members = 0;
            member->nested_members = NULL;
        }

        // 解析offset
        int offset = 0;
        while (*sig_str >= '0' && *sig_str <= '9') {
            offset = offset * 10 + (*sig_str - '0');
            sig_str++;
        }
        member->offset = offset;

        if (*sig_str == ',') {
            sig_str++;
        } else if (*sig_str == '}') {
            break;
        }
    }
    if (*sig_str == '}') sig_str++;
    return sig_str;
}

// Parses a full parameter signature string: name:type_info
static void parse_param_signature(char* sig_str, ParamInfoV2* param) {
    char* type_info_str = strchr(sig_str, ':');
    if (type_info_str) {
        *type_info_str = '\0';
        strncpy(param->name, sig_str, sizeof(param->name) - 1);
        param->name[sizeof(param->name) - 1] = '\0'; // Ensure null termination
        parse_type_info(type_info_str + 1, param);
    }
}

// --- Runtime API Functions ---

void __cuda_profile_kernel_start(const char* kernel_name,
                                 int gridx, int gridy, int gridz,
                                 int blockx, int blocky, int blockz) {
    if (!profiling_enabled) return;
    fprintf(stderr, "[CudaArgsProfile] Launching kernel: %s\n", kernel_name);

    memset(&current_kernel, 0, sizeof(current_kernel));
    current_kernel.id = launch_count;
    strncpy(current_kernel.name, kernel_name, sizeof(current_kernel.name) - 1);
    current_kernel.name[sizeof(current_kernel.name) - 1] = '\0'; // Ensure null termination
    current_kernel.grid[0] = gridx; current_kernel.grid[1] = gridy; current_kernel.grid[2] = gridz;
    current_kernel.block[0] = blockx; current_kernel.block[1] = blocky; current_kernel.block[2] = blockz;
}

// Called for each argument before kernel launch
void __cuda_profile_argument(int index, const char* signature, void* value_ptr) {
    if (!profiling_enabled) return;
    if (index >= MAX_PARAMS) return;

    // We need to parse the size from the signature to copy the value
    char* sig_copy_for_size = strdup(signature);
    if (!sig_copy_for_size) return;
    ParamInfoV2 temp_param = {0};
    parse_param_signature(sig_copy_for_size, &temp_param);
    free(sig_copy_for_size);

    if (temp_param.size <= 0) return;

    void* value_copy = malloc(temp_param.size);
    if (!value_copy) return;
    memcpy(value_copy, value_ptr, temp_param.size);

    fprintf(stderr, "[CudaArgsProfile] Got argument index %d, value_ptr %p, size %d\n", index, value_ptr, temp_param.size);

    current_kernel.arg_values[index] = value_copy;
    current_kernel.arg_signatures[index] = strdup(signature);
    current_kernel.num_params = index + 1;
}

void __cuda_profile_kernel_end(void) {
    if (!profiling_enabled) return;
    if (launch_count < MAX_KERNELS) {
        kernel_launches[launch_count++] = current_kernel;
    }
    fprintf(stderr, "[CudaArgsProfile] Kernel launch ended.\n");
}

// --- JSON Generation ---

static void print_json_value(FILE* fp, const ParamInfoV2* param, void* data_ptr);
static void print_json_members(FILE* fp, const MemberInfoV2* members, int num_members, void* base_ptr);

static void print_json_value(FILE* fp, const ParamInfoV2* param, void* data_ptr) {
    if (!data_ptr) {
        fprintf(fp, "\"null\"");
        return;
    }

    switch (param->type_kind) {
        case 's': { // scalar
            if (strcmp(param->type_name, "float") == 0) {
                fprintf(fp, "%f", *(float*)data_ptr);
            } else if (strcmp(param->type_name, "double") == 0) {
                fprintf(fp, "%lf", *(double*)data_ptr);
            } else { // treat as integer
                long long val = 0;
                if (param->size == 1) val = *(int8_t*)data_ptr;
                else if (param->size == 2) val = *(int16_t*)data_ptr;
                else if (param->size == 4) val = *(int32_t*)data_ptr;
                else if (param->size == 8) val = *(int64_t*)data_ptr;
                fprintf(fp, "%lld", val);
            }
            break;
        }
        case 'p': { // pointer
            fprintf(fp, "\"0x%lx\"", (unsigned long)*(uintptr_t*)data_ptr);
            break;
        }
        case 'r': { // record/struct
            fprintf(fp, "{");
            print_json_members(fp, param->members, param->num_members, data_ptr);
            fprintf(fp, "}");
            break;
        }
        default:
            fprintf(fp, "\"unknown type: %c\"", param->type_kind);
    }
}

static void free_param_info_members(ParamInfoV2* param) {
    if (!param) return;
    for (int i = 0; i < param->num_members; ++i) {
        if (param->members[i].nested_members) {
            free_param_info_members(param->members[i].nested_members);
            free(param->members[i].nested_members);
        }
    }
}

static void print_json_members(FILE* fp, const MemberInfoV2* members, int num_members, void* base_ptr) {
    for (int i = 0; i < num_members; ++i) {
        const MemberInfoV2* member = &members[i];
        fprintf(fp, "\"%s\": { \"type\": \"%s\", \"value\": ", member->name, member->type_name); // Include type

        void* member_ptr = ((char*)base_ptr) + member->offset;

        // Create a temporary ParamInfoV2 to reuse print_json_value
        ParamInfoV2 member_as_param;
        strncpy(member_as_param.name, member->name, sizeof(member_as_param.name)-1);
        member_as_param.name[sizeof(member_as_param.name)-1] = '\0'; // Ensure null termination
        strncpy(member_as_param.type_name, member->type_name, sizeof(member_as_param.type_name)-1);
        member_as_param.type_name[sizeof(member_as_param.type_name)-1] = '\0'; // Ensure null termination
        member_as_param.type_kind = member->type_kind;
        member_as_param.size = member->size;
        if (member->type_kind == 'r') {
            member_as_param.num_members = member->num_nested_members;
            // This is a simplification; deep nesting requires more care
            if (member->nested_members) {
                 memcpy(member_as_param.members, member->nested_members->members, sizeof(MemberInfoV2) * member->num_nested_members);
            }
        } else {
            member_as_param.num_members = 0;
        }

        print_json_value(fp, &member_as_param, member_ptr);

        fprintf(fp, "}"); // Close the member object

        if (i < num_members - 1) {
            fprintf(fp, ", ");
        }
    }
}

void __cuda_profile_finalize(void) {
    if (!profiling_enabled) return;

    FILE* fp = fopen("cuda_profile.json", "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open cuda_profile.json for writing\n");
        return;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"launches\": [\n");

    for (int i = 0; i < launch_count; ++i) {
        KernelLaunch* launch = &kernel_launches[i];
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"id\": %d,\n", launch->id);
        fprintf(fp, "      \"name\": \"%s\",\n", launch->name);
        fprintf(fp, "      \"grid\": [%d,%d,%d],\n", launch->grid[0], launch->grid[1], launch->grid[2]);
        fprintf(fp, "      \"block\": [%d,%d,%d],\n", launch->block[0], launch->block[1], launch->block[2]);
        fprintf(fp, "      \"params\": [\n");

        for (int j = 0; j < launch->num_params; ++j) {
            char* sig_copy = strdup(launch->arg_signatures[j]);
            ParamInfoV2 param = {0};
            if (sig_copy) {
                parse_param_signature(sig_copy, &param);
                free(sig_copy);
            }

            fprintf(fp, "        {\n");
            fprintf(fp, "          \"name\": \"%s\",\n", param.name);
            fprintf(fp, "          \"type\": \"%s\",\n", param.type_name);
            fprintf(fp, "          \"size\": %d,\n", param.size);
            fprintf(fp, "          \"value\": ");
            print_json_value(fp, &param, launch->arg_values[j]);
            fprintf(fp, "\n");
            fprintf(fp, "        }");
            if (j < launch->num_params - 1) {
                fprintf(fp, ",\n");
            }
            
            // Free copied data
            if(launch->arg_signatures[j]) free(launch->arg_signatures[j]);
            if(launch->arg_values[j]) free(launch->arg_values[j]);
            free_param_info_members(&param);
        }

        fprintf(fp, "\n      ]\n");
        fprintf(fp, "    }");
        if (i < launch_count - 1) {
            fprintf(fp, ",\n");
        }
    }

    fprintf(fp, "\n  ]\n");
    fprintf(fp, "}\n");
    fclose(fp);

    printf("CUDA arguments profiling results written to cuda_profile.json\n");
}