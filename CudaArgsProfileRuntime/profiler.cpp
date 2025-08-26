#include "profiler.h"
#include "process_profile.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <regex>
#include "json.hpp"
#include <sys/file.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <dlfcn.h>

using json = nlohmann::json;

// Global JSON object to store all kernel launch data
static json profile_data;
static std::mutex profile_data_mutex;

// This function will be called at exit to write the file.
void write_profile_data() {
    const char* json_path_env = std::getenv("CUDA_ARGS_PROFILE_JSON_FILE");
    if (!json_path_env) return;
    std::string json_path = json_path_env;

    const char* append_env = std::getenv("CUDA_ARGS_PROFILE_APPEND");
    bool append = (append_env && std::string(append_env) == "1");

    json final_root;

    int fd = open(json_path.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
        std::cerr << "Profiler: Error opening or creating file: " << json_path << std::endl;
        return;
    }
    if (flock(fd, LOCK_EX) == -1) {
        std::cerr << "Profiler: Error locking file: " << json_path << std::endl;
        close(fd);
        return;
    }

    if (append) {
        std::ifstream read_file(json_path);
        if (read_file.peek() != std::ifstream::traits_type::eof()) {
            try {
                final_root = json::parse(read_file, nullptr, false);
                if (final_root.is_discarded()) { // Handle parse error
                  final_root = json::object();
                }
            } catch (json::parse_error& e) {
                final_root = json::object();
            }
        }
        read_file.close();
    }

    std::lock_guard<std::mutex> lock(profile_data_mutex);
    if (!final_root.contains("kernels") || !final_root["kernels"].is_array()) {
        final_root["kernels"] = json::array();
    }
    for (const auto& launch : profile_data["kernels"]) {
        final_root["kernels"].push_back(launch);
    }

    std::ofstream write_file(json_path, std::ios::trunc);
    write_file << final_root.dump(2);
    write_file.close();

    flock(fd, LOCK_UN);
    close(fd);

    process_profile_data();
}

namespace {

struct ProfilerExitHandler {
    ProfilerExitHandler() {
        atexit(write_profile_data);
    }
};

ProfilerExitHandler exit_handler;

} // namespace

// 保存原始的CUDA函数指针
static cudaError_t (*original_cudaMalloc)(void **, size_t) = nullptr;
static cudaError_t (*original_cudaFree)(void *) = nullptr;
static cudaError_t (*original_cudaMemcpy)(void *, const void *, size_t, cudaMemcpyKind) = nullptr;
static cudaError_t (*original_cudaMemcpyAsync)(void *, const void *, size_t, cudaMemcpyKind, cudaStream_t) = nullptr;
static cudaError_t (*original_cudaMallocHost)(void **, size_t) = nullptr;
static cudaError_t (*original_cudaFreeHost)(void *) = nullptr;
static cudaError_t (*original_cudaMallocManaged)(void **, size_t, unsigned int) = nullptr;

// 内存区域信息结构
struct MemoryRegion {
    void* base_addr;
    size_t size;
    std::string type;
    
    MemoryRegion(void* addr, size_t sz, const std::string& t) 
        : base_addr(addr), size(sz), type(t) {}
    
    // 检查指针是否在这个内存区域内
    bool contains(void* ptr) const {
        return ptr >= base_addr && 
               ptr < (char*)base_addr + size;
    }
    
    // 计算指针在这个区域内的偏移量
    size_t getOffset(void* ptr) const {
        return (char*)ptr - (char*)base_addr;
    }
    
    // 计算从指针位置开始的剩余大小
    size_t getRemainingSize(void* ptr) const {
        if (!contains(ptr)) return 0;
        return size - getOffset(ptr);
    }
};

// 存储所有内存区域信息
std::vector<MemoryRegion> memory_regions;
// 为了快速查找，也保留原来的map（用于精确匹配）
std::unordered_map<void*, size_t> memory_map;
std::unordered_map<void*, std::string> memory_type_map;

// 初始化函数，获取原始CUDA函数指针
static void init_original_functions() {
    if (!original_cudaMalloc) {
        original_cudaMalloc = (cudaError_t (*)(void **, size_t))dlsym(RTLD_NEXT, "cudaMalloc");
        if (!original_cudaMalloc) {
            std::cerr << "Failed to get original cudaMalloc function" << std::endl;
        }
    }
    
    if (!original_cudaFree) {
        original_cudaFree = (cudaError_t (*)(void *))dlsym(RTLD_NEXT, "cudaFree");
        if (!original_cudaFree) {
            std::cerr << "Failed to get original cudaFree function" << std::endl;
        }
    }
    
    if (!original_cudaMemcpy) {
        original_cudaMemcpy = (cudaError_t (*)(void *, const void *, size_t, cudaMemcpyKind))dlsym(RTLD_NEXT, "cudaMemcpy");
        if (!original_cudaMemcpy) {
            std::cerr << "Failed to get original cudaMemcpy function" << std::endl;
        }
    }
    
    if (!original_cudaMemcpyAsync) {
        original_cudaMemcpyAsync = (cudaError_t (*)(void *, const void *, size_t, cudaMemcpyKind, cudaStream_t))dlsym(RTLD_NEXT, "cudaMemcpyAsync");
        if (!original_cudaMemcpyAsync) {
            std::cerr << "Failed to get original cudaMemcpyAsync function" << std::endl;
        }
    }
    
    if (!original_cudaMallocHost) {
        original_cudaMallocHost = (cudaError_t (*)(void **, size_t))dlsym(RTLD_NEXT, "cudaMallocHost");
        if (!original_cudaMallocHost) {
            std::cerr << "Failed to get original cudaMallocHost function" << std::endl;
        }
    }
    
    if (!original_cudaFreeHost) {
        original_cudaFreeHost = (cudaError_t (*)(void *))dlsym(RTLD_NEXT, "cudaFreeHost");
        if (!original_cudaFreeHost) {
            std::cerr << "Failed to get original cudaFreeHost function" << std::endl;
        }
    }
    
    if (!original_cudaMallocManaged) {
        original_cudaMallocManaged = (cudaError_t (*)(void **, size_t, unsigned int))dlsym(RTLD_NEXT, "cudaMallocManaged");
        if (!original_cudaMallocManaged) {
            std::cerr << "Failed to get original cudaMallocManaged function" << std::endl;
        }
    }
}

// Hook cudaMalloc函数
extern "C" cudaError_t cudaMalloc(void **devPtr, size_t size) {
    init_original_functions();
    
    if (!original_cudaMalloc) {
        return cudaErrorUnknown;
    }
    
    cudaError_t err = original_cudaMalloc(devPtr, size);
    if (err == cudaSuccess) {
        memory_map[*devPtr] = size;
        memory_type_map[*devPtr] = "device";
        memory_regions.emplace_back(*devPtr, size, "device");
    }
    return err;
}

// Hook cudaFree函数
extern "C" cudaError_t cudaFree(void *devPtr) {
    init_original_functions();
    
    if (!original_cudaFree) {
        return cudaErrorUnknown;
    }
    
    cudaError_t err = original_cudaFree(devPtr);
    if (err == cudaSuccess) {
        memory_map.erase(devPtr);
        memory_type_map.erase(devPtr);
        // 从memory_regions中移除对应的区域
        memory_regions.erase(
            std::remove_if(memory_regions.begin(), memory_regions.end(),
                [devPtr](const MemoryRegion& region) {
                    return region.base_addr == devPtr;
                }),
            memory_regions.end()
        );
    }
    return err;
}

// Hook cudaMemcpy函数
extern "C" cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
    init_original_functions();
    
    if (!original_cudaMemcpy) {
        return cudaErrorUnknown;
    }
    
    // 记录内存拷贝信息（可选）
    // 这里可以添加内存拷贝的统计信息
    
    return original_cudaMemcpy(dst, src, count, kind);
}

// Hook cudaMemcpyAsync函数
extern "C" cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    init_original_functions();
    
    if (!original_cudaMemcpyAsync) {
        return cudaErrorUnknown;
    }
    
    return original_cudaMemcpyAsync(dst, src, count, kind, stream);
}

// Hook cudaMallocHost函数
extern "C" cudaError_t cudaMallocHost(void **ptr, size_t size) {
    init_original_functions();
    
    if (!original_cudaMallocHost) {
        return cudaErrorUnknown;
    }
    
    cudaError_t err = original_cudaMallocHost(ptr, size);
    if (err == cudaSuccess) {
        memory_map[*ptr] = size;
        memory_type_map[*ptr] = "host";
        memory_regions.emplace_back(*ptr, size, "host");
    }
    return err;
}

// Hook cudaFreeHost函数
extern "C" cudaError_t cudaFreeHost(void *ptr) {
    init_original_functions();
    
    if (!original_cudaFreeHost) {
        return cudaErrorUnknown;
    }
    
    cudaError_t err = original_cudaFreeHost(ptr);
    if (err == cudaSuccess) {
        memory_map.erase(ptr);
        memory_type_map.erase(ptr);
        // 从memory_regions中移除对应的区域
        memory_regions.erase(
            std::remove_if(memory_regions.begin(), memory_regions.end(),
                [ptr](const MemoryRegion& region) {
                    return region.base_addr == ptr;
                }),
            memory_regions.end()
        );
    }
    return err;
}

// Hook cudaMallocManaged函数
extern "C" cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags) {
    init_original_functions();
    
    if (!original_cudaMallocManaged) {
        return cudaErrorUnknown;
    }
    
    cudaError_t err = original_cudaMallocManaged(devPtr, size, flags);
    if (err == cudaSuccess) {
        memory_map[*devPtr] = size;
        memory_type_map[*devPtr] = "managed";
        memory_regions.emplace_back(*devPtr, size, "managed");
    }
    return err;
}

// 查找指针所属的内存区域
const MemoryRegion* findMemoryRegion(void* ptr) {
    for (const auto& region : memory_regions) {
        if (region.contains(ptr)) {
            return &region;
        }
    }
    return nullptr;
}

// Forward declaration
void parse_and_add_value(json& param_info, char* data_addr);

// New helper to parse array types like "int[3]"
bool parse_array_type(const std::string& type, std::string& base_type, int& count) {
    static std::regex re(R"((.*?)\s*\[\s*(\d+)\s*\])");
    std::smatch match;
    if (std::regex_match(type, match, re)) {
        base_type = match[1].str();
        count = std::stoi(match[2].str());
        return true;
    }
    return false;
}

// helper to get scalar values of various types
json get_scalar_value(const std::string& type, char* data_addr) {
    if (type == "int") return *reinterpret_cast<int*>(data_addr);
    if (type == "float") return *reinterpret_cast<float*>(data_addr);
    if (type == "double") return *reinterpret_cast<double*>(data_addr);
    if (type == "char") return *reinterpret_cast<char*>(data_addr);
    if (type == "short") return *reinterpret_cast<short*>(data_addr);
    if (type == "long") return *reinterpret_cast<long*>(data_addr);
    if (type == "long long") return *reinterpret_cast<long long*>(data_addr);
    if (type == "unsigned char") return *reinterpret_cast<unsigned char*>(data_addr);
    if (type == "unsigned short") return *reinterpret_cast<unsigned short*>(data_addr);
    if (type == "unsigned int") return *reinterpret_cast<unsigned int*>(data_addr);
    if (type == "unsigned long") return *reinterpret_cast<unsigned long*>(data_addr);
    if (type == "unsigned long long") return *reinterpret_cast<unsigned long long*>(data_addr);
    if (type == "bool" || type == "_Bool") return *reinterpret_cast<bool*>(data_addr);
    return "unsupported_scalar_type";
}

// Parses members of a struct, creating a new JSON array for the "value"
void parse_struct_members(json& members_array, char* struct_base_addr) {
    for (auto& member_info : members_array) {
        size_t offset = member_info["offset"].get<size_t>();
        parse_and_add_value(member_info, struct_base_addr + offset);
    }
}

// Main recursive parsing function. It takes a JSON object containing the type info
// and adds a "value" field to it.
void parse_and_add_value(json& param_info, char* data_addr) {
    std::string type = param_info["type"].get<std::string>();
    size_t size = param_info.value("size", 0);

    std::string base_type;
    int count;

    if (parse_array_type(type, base_type, count)) {
        // It's an array
        json arr = json::array();
        if (count > 0) {
            size_t element_size = size / count;
            if (element_size > 0) {
                for (int i = 0; i < count; ++i) {
                    arr.push_back(get_scalar_value(base_type, data_addr + i * element_size));
                }
            }
        }
        param_info["value"] = arr;
    } else if (param_info.contains("members")) {
        // It's a struct
        json members_copy = param_info["members"];
        parse_struct_members(members_copy, data_addr);
        param_info["value"] = members_copy;
        param_info.erase("members");
    } else if (type.find('*') != std::string::npos) {
        // It's a pointer
        void* ptr_value = *reinterpret_cast<void**>(data_addr);
        char hex_buf[20];
        sprintf(hex_buf, "%p", ptr_value);
        param_info["value"] = hex_buf;

        const MemoryRegion* region = findMemoryRegion(ptr_value);
        if (region) {
            param_info["size"] = region->getRemainingSize(ptr_value);
            param_info["memory_type"] = region->type;
            char base_addr_buf[20];
            sprintf(base_addr_buf, "%p", region->base_addr);
            param_info["base_address"] = base_addr_buf;
            param_info["offset"] = region->getOffset(ptr_value);
        }
    } else {
        // It's a scalar
        param_info["value"] = get_scalar_value(type, data_addr);
    }
}

extern "C" void __cuda_profile_kernel_launch(const char* kernel_name, const char* device_side_name, int arg_count, void** arg_values, const char* arg_info_json_str, void* grid_dim, void* block_dim) {
    std::lock_guard<std::mutex> lock(profile_data_mutex);
    if (!profile_data.contains("kernels") || !profile_data["kernels"].is_array()) {
        profile_data["kernels"] = json::array();
    }

    json kernel_launch_info = json::parse(arg_info_json_str, nullptr, false);
     if (kernel_launch_info.is_discarded()) {
        std::cerr << "Profiler: Failed to parse kernel info JSON" << std::endl;
        return;
    }

    kernel_launch_info["id"] = profile_data["kernels"].size();
    kernel_launch_info["device_side_name"] = device_side_name;

    json params_with_values = json::array();
    if(kernel_launch_info.contains("params") && kernel_launch_info["params"].is_array()){
        for (int i = 0; i < arg_count; ++i) {
            json param_info = kernel_launch_info["params"][i];
            parse_and_add_value(param_info, static_cast<char*>(arg_values[i]));
            params_with_values.push_back(param_info);
        }
    }

    // 计算同一次launch中顶层指针参数的别名关系（区间重叠即视为别名）
    {
        struct PtrRange { size_t idx; unsigned long long start; unsigned long long end; };
        std::vector<PtrRange> ptrs;
        ptrs.reserve(params_with_values.size());

        for (size_t i = 0; i < params_with_values.size(); ++i) {
            const auto& p = params_with_values[i];
            if (!p.contains("type") || !p["type"].is_string()) continue;
            const std::string t = p["type"].get<std::string>();
            if (t.find('*') == std::string::npos) continue; // 仅顶层指针参数

            // 需要有地址和size
            if (!p.contains("value") || !p["value"].is_string()) continue;
            if (!p.contains("size") || !p["size"].is_number_unsigned()) continue;

            const std::string addr_str = p["value"].get<std::string>();
            if (addr_str.rfind("0x", 0) != 0) continue;
            unsigned long long addr = 0ULL;
            try {
                addr = std::stoull(addr_str, nullptr, 16);
            } catch (...) {
                continue;
            }
            unsigned long long sz = p["size"].get<unsigned long long>();
            if (sz == 0ULL) continue;
            ptrs.push_back(PtrRange{(size_t)i, addr, addr + sz});
        }

        // 默认alias=0，然后若检测到与任一其它指针重叠则置为1
        for (size_t i = 0; i < params_with_values.size(); ++i) {
            auto& p = params_with_values[i];
            if (p.contains("type") && p["type"].is_string() && p["type"].get<std::string>().find('*') != std::string::npos) {
                p["alias"] = 0;
            }
        }

        for (size_t i = 0; i < ptrs.size(); ++i) {
            for (size_t j = i + 1; j < ptrs.size(); ++j) {
                const auto& a = ptrs[i];
                const auto& b = ptrs[j];
                // 区间[a.start, a.end) 与 [b.start, b.end) 是否重叠
                if (std::max(a.start, b.start) < std::min(a.end, b.end)) {
                    params_with_values[a.idx]["alias"] = 1;
                    params_with_values[b.idx]["alias"] = 1;
                }
            }
        }
    }

    kernel_launch_info["params"] = params_with_values;

    dim3* grid = static_cast<dim3*>(grid_dim);
    kernel_launch_info["grid"] = {grid->x, grid->y, grid->z};
    dim3* block = static_cast<dim3*>(block_dim);
    kernel_launch_info["block"] = {block->x, block->y, block->z};

    profile_data["kernels"].push_back(kernel_launch_info);
}