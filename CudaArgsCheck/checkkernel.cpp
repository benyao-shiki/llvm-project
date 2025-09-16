#include <dlfcn.h>
#include <cuda.h>
#include <vector>
#include <utility>
#include <mutex>
#include <iostream>
#include <iomanip>
#include <algorithm>

// Data structures to track memory allocations
struct MemoryRegion {
    CUdeviceptr base_addr;
    size_t size;

    MemoryRegion(CUdeviceptr addr, size_t s) : base_addr(addr), size(s) {}

    bool contains(CUdeviceptr ptr) const {
        return ptr >= base_addr && ptr < (base_addr + size);
    }
};

static std::vector<MemoryRegion> g_memory_regions;
static std::mutex g_mutex;

// Pointers to original CUDA driver functions
static CUresult (*original_cuMemAlloc_v2)(CUdeviceptr*, size_t) = nullptr;
static CUresult (*original_cuMemFree_v2)(CUdeviceptr) = nullptr;
static CUresult (*original_cuMemAllocManaged)(CUdeviceptr*, size_t, unsigned int) = nullptr;

// Initialization function to get original function pointers
static void init_original_functions() {
    static bool initialized = false;
    if (initialized) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (initialized) return;

    original_cuMemAlloc_v2 = (CUresult (*)(CUdeviceptr*, size_t))dlsym(RTLD_NEXT, "cuMemAlloc_v2");
    if (!original_cuMemAlloc_v2) {
        std::cerr << "Failed to get original cuMemAlloc_v2 function" << std::endl;
    }

    original_cuMemFree_v2 = (CUresult (*)(CUdeviceptr))dlsym(RTLD_NEXT, "cuMemFree_v2");
    if (!original_cuMemFree_v2) {
        std::cerr << "Failed to get original cuMemFree_v2 function" << std::endl;
    }
    
    original_cuMemAllocManaged = (CUresult (*)(CUdeviceptr*, size_t, unsigned int))dlsym(RTLD_NEXT, "cuMemAllocManaged");
    if (!original_cuMemAllocManaged) {
        std::cerr << "Failed to get original cuMemAllocManaged function" << std::endl;
    }

    initialized = true;
}

// Hooked cuMemAlloc_v2
extern "C" CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
    init_original_functions();
    if (!original_cuMemAlloc_v2) return CUDA_ERROR_UNKNOWN;

    CUresult err = original_cuMemAlloc_v2(dptr, bytesize);
    if (err == CUDA_SUCCESS) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_memory_regions.emplace_back(*dptr, bytesize);
    }
    return err;
}

// Hooked cuMemFree_v2
extern "C" CUresult cuMemFree_v2(CUdeviceptr dptr) {
    init_original_functions();
    if (!original_cuMemFree_v2) return CUDA_ERROR_UNKNOWN;

    CUresult err = original_cuMemFree_v2(dptr);
    if (err == CUDA_SUCCESS) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_memory_regions.erase(
            std::remove_if(g_memory_regions.begin(), g_memory_regions.end(),
                [dptr](const MemoryRegion& region) {
                    return region.base_addr == dptr;
                }),
            g_memory_regions.end()
        );
    }
    return err;
}

// Hooked cuMemAllocManaged
extern "C" CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags) {
    init_original_functions();
    if (!original_cuMemAllocManaged) return CUDA_ERROR_UNKNOWN;

    CUresult err = original_cuMemAllocManaged(dptr, bytesize, flags);
    if (err == CUDA_SUCCESS) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_memory_regions.emplace_back(*dptr, bytesize);
    }
    return err;
}


// Helper to find the memory region for a given pointer
const MemoryRegion* findMemoryRegion(CUdeviceptr ptr) {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (const auto& region : g_memory_regions) {
        if (region.contains(ptr)) {
            return &region;
        }
    }
    return nullptr;
}

extern "C" {

/*
 * check_ptr_sets
 *    num_targets : size of first pointer array (targets)
 *    targets     : void** array containing pointers to be individually checked
 *    num_others  : size of second pointer array (others)
 *    others      : void** array containing pointers to compare against
 *
 * Returns true  if any target pointer overlaps with any other pointer (excluding itself)
 *         false otherwise
 */
bool check_ptr_sets(int num_targets, void* const* targets,
                    int num_others,  void* const* others) {
    if (num_targets == 0) return false;

    auto get_range = [](void* p) -> std::pair<CUdeviceptr, size_t> {
        const MemoryRegion* region = findMemoryRegion((CUdeviceptr)p);
        if (region) {
            return std::make_pair(region->base_addr, region->size);
        }
        // If not found in our tracked allocations, return a zero-sized range
        return std::make_pair((CUdeviceptr)p, 0);
    };

    std::vector<std::pair<CUdeviceptr, size_t>> targ_ranges(num_targets);
    std::vector<std::pair<CUdeviceptr, size_t>> oth_ranges(num_others);

    for(int i = 0; i < num_targets; ++i) targ_ranges[i] = get_range(targets[i]);
    for(int i = 0; i < num_others; ++i) oth_ranges[i] = get_range(others[i]);

    for(int i = 0; i < num_targets; ++i) {
        CUdeviceptr a0 = targ_ranges[i].first;
        size_t a_size = targ_ranges[i].second;
        if (a_size == 0) continue;
        CUdeviceptr a1 = a0 + a_size;

        for(int j = 0; j < num_others; ++j) {
            CUdeviceptr b0 = oth_ranges[j].first;
            size_t b_size = oth_ranges[j].second;
            if (b_size == 0) continue;
            CUdeviceptr b1 = b0 + b_size;
            if (a0 < b1 && b0 < a1) return true;
        }

        for(int k = 0; k < num_targets; ++k) {
            if (k == i) continue;
            CUdeviceptr b0 = targ_ranges[k].first;
            size_t b_size = targ_ranges[k].second;
            if (b_size == 0) continue;
            CUdeviceptr b1 = b0 + b_size;
            if (a0 < b1 && b0 < a1) return true;
        }
    }
    return false;
}


/*
 * check_const
 *    n_values : number of elements in both arrays
 *    expected : array of expected int64_t values
 *    actual   : array of actual int64_t values
 *
 * Returns true  if all elements in actual match the corresponding elements in expected
 *         false if any element differs
 */
 bool check_const(int n_values, int64_t *expected, int64_t *actual) {
    if (n_values <= 0) return true;
    for (int i = 0; i < n_values; ++i) {
        if (expected[i] != actual[i]) {
            std::cout << "check_const: false at index " << i << " expected: " << expected[i] << " actual: " << actual[i] << std::endl;

            // --- BEGIN DEBUG PATCH ---
            // Print the raw hex value of actual to understand its composition
            unsigned char* p = reinterpret_cast<unsigned char*>(&actual[i]);
            std::cout << "check_const: actual[" << i << "] raw hex (8 bytes): ";
            for (int j = 0; j < 8; ++j) {
                // Print as 2-digit hex
                std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)p[j] << " ";
            }
            std::cout << std::dec << std::endl;
            // --- END DEBUG PATCH ---

            return false;
        }
    }
    return true;
}

}
