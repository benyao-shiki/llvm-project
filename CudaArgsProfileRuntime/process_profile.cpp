#include "process_profile.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <map>
#include "json.hpp"

using json = nlohmann::json;

// Control variables with default values
static const int MIN_LAUNCH_COUNT = std::getenv("HOT_KERNELS_MIN_LAUNCH_COUNT") ? std::stoi(std::getenv("HOT_KERNELS_MIN_LAUNCH_COUNT")) : 1;
static const double MIN_NOALIAS_RATIO = std::getenv("HOT_KERNELS_MIN_NOALIAS_RATIO") ? std::stod(std::getenv("HOT_KERNELS_MIN_NOALIAS_RATIO")) : 0.6;
static const double MIN_COMMON_SCALAR_RATIO = std::getenv("HOT_KERNELS_MIN_COMMON_SCALAR_RATIO") ? std::stod(std::getenv("HOT_KERNELS_MIN_COMMON_SCALAR_RATIO")) : 0.6;

struct PointerInfo {
    uintptr_t address;
    size_t size;
};

bool do_pointers_alias(const PointerInfo& p1, const PointerInfo& p2) {
    if (p1.address == 0 || p2.address == 0) return false;
    uintptr_t start1 = p1.address;
    uintptr_t end1 = p1.address + p1.size;
    uintptr_t start2 = p2.address;
    uintptr_t end2 = p2.address + p2.size;
    return std::max(start1, start2) < std::min(end1, end2);
}

// --- Common Scalars ---
struct ScalarIdentifier {
    int param_index;
    int member_index = -1; // -1 indicates not a member
    std::string name;

    bool operator<(const ScalarIdentifier& other) const {
        if (param_index != other.param_index) {
            return param_index < other.param_index;
        }
        return member_index < other.member_index;
    }
};

void find_common_scalars_recursive(const json& param, const std::string& name_prefix, int param_index, int member_index, std::map<ScalarIdentifier, std::map<json, int>>& scalar_counts) {
    std::string current_name = name_prefix + param["name"].get<std::string>();
    if (param.contains("value") && param["value"].is_array()) { // It's a struct
        for (size_t i = 0; i < param["value"].size(); ++i) {
            find_common_scalars_recursive(param["value"][i], current_name + ".", param_index, i, scalar_counts);
        }
    } else if (param["type"].get<std::string>().find('*') == std::string::npos) { // It's a scalar
        ScalarIdentifier id;
        id.param_index = param_index;
        id.member_index = member_index;
        id.name = current_name;
        scalar_counts[id][param["value"]]++;
    }
}

void find_common_scalars(const std::vector<json>& launches, json& hot_kernel_info) {
    if (launches.empty()) return;
    std::map<ScalarIdentifier, std::map<json, int>> scalar_counts;
    std::map<std::string, std::map<json, int>> grid_block_counts;


    for (const auto& launch : launches) {
        for (size_t i = 0; i < launch["params"].size(); ++i) {
            find_common_scalars_recursive(launch["params"][i], "", i, -1, scalar_counts);
        }
        grid_block_counts["grid.x"][launch["grid"][0]]++;
        grid_block_counts["grid.y"][launch["grid"][1]]++;
        grid_block_counts["grid.z"][launch["grid"][2]]++;
        grid_block_counts["block.x"][launch["block"][0]]++;
        grid_block_counts["block.y"][launch["block"][1]]++;
        grid_block_counts["block.z"][launch["block"][2]]++;
    }

    json common_scalars = json::array();
    for (auto const& [id, counts] : scalar_counts) {
        for (auto const& [value, count] : counts) {
            double ratio = static_cast<double>(count) / launches.size();
            if (ratio >= MIN_COMMON_SCALAR_RATIO) {
                json scalar_info;
                scalar_info["name"] = id.name;
                scalar_info["value"] = value;
                scalar_info["ratio"] = ratio;
                scalar_info["param_index"] = id.param_index;
                if (id.member_index != -1) {
                    scalar_info["member_index"] = id.member_index;
                }
                common_scalars.push_back(scalar_info);
            }
        }
    }

    for (auto const& [name, counts] : grid_block_counts) {
        for (auto const& [value, count] : counts) {
            double ratio = static_cast<double>(count) / launches.size();
            if (ratio >= MIN_COMMON_SCALAR_RATIO) {
                json scalar_info;
                scalar_info["name"] = name;
                scalar_info["value"] = value;
                scalar_info["ratio"] = ratio;
                common_scalars.push_back(scalar_info);
            }
        }
    }

    if (!common_scalars.empty()) {
        hot_kernel_info["common_scalars"] = common_scalars;
    }
}


// --- No-alias Pointers ---
struct PointerParameter {
    std::string name;
    int top_level_index;
    int member_index = -1; // -1 indicates not a member
    uintptr_t address;
    size_t size;
};

void find_pointers_recursive(const json& param, const std::string& name_prefix, int top_level_index, int member_index, std::vector<PointerParameter>& pointers) {
    std::string current_name = name_prefix + param["name"].get<std::string>();
    if (param.contains("value") && param["value"].is_array()) { // It's a struct
        for (size_t i = 0; i < param["value"].size(); ++i) {
            find_pointers_recursive(param["value"][i], current_name + ".", top_level_index, i, pointers);
        }
    } else if (param["type"].get<std::string>().find('*') != std::string::npos) { // It's a pointer
        PointerParameter p;
        p.name = current_name;
        p.top_level_index = top_level_index;
        p.member_index = member_index;
        if (param["value"].is_string()) {
            std::string val_str = param["value"].get<std::string>();
            if (val_str.rfind("0x", 0) == 0) {
                p.address = std::stoull(val_str, nullptr, 16);
                p.size = param.value("size", 0);
                pointers.push_back(p);
            }
        }
    }
}

void find_noalias_pointers(const std::vector<json>& launches, json& hot_kernel_info) {
    if (launches.empty()) return;

    std::vector<std::vector<PointerParameter>> launch_pointers;
    for(const auto& launch : launches) {
        std::vector<PointerParameter> current_launch_pointers;
        const auto& params = launch["params"];
        for (size_t i = 0; i < params.size(); ++i) {
            find_pointers_recursive(params[i], "", i, -1, current_launch_pointers);
        }
        launch_pointers.push_back(current_launch_pointers);
    }

    if (launch_pointers.empty() || launch_pointers.front().empty()) {
        return;
    }

    const auto& first_launch_ptrs = launch_pointers.front();
    json noalias_pointers = json::array();

    if (first_launch_ptrs.size() == 1) {
        const auto& ptr = first_launch_ptrs[0];
        json ptr_info;
        ptr_info["param_name"] = ptr.name;
        ptr_info["param_index"] = ptr.top_level_index;
        if (ptr.member_index != -1) {
            ptr_info["member_index"] = ptr.member_index;
        }
        ptr_info["noalias_ratio"] = 1.0;
        noalias_pointers.push_back(ptr_info);
    } else if (first_launch_ptrs.size() > 1) {
        std::map<std::pair<int, int>, int> noalias_counts;
        for (size_t i = 0; i < first_launch_ptrs.size(); ++i) {
            for (size_t j = i + 1; j < first_launch_ptrs.size(); ++j) {
                noalias_counts[{i, j}] = 0;
            }
        }

        for (const auto& current_launch_pointers : launch_pointers) {
            for (auto it = noalias_counts.begin(); it != noalias_counts.end(); ++it) {
                int i = it->first.first;
                int j = it->first.second;

                PointerInfo p1 = {current_launch_pointers[i].address, current_launch_pointers[i].size};
                PointerInfo p2 = {current_launch_pointers[j].address, current_launch_pointers[j].size};

                if (!do_pointers_alias(p1, p2)) {
                    it->second++;
                }
            }
        }

        for (auto const& [key, val] : noalias_counts) {
            double ratio = static_cast<double>(val) / launches.size();
            if (ratio >= MIN_NOALIAS_RATIO) {
                const auto& ptr1 = first_launch_ptrs[key.first];
                const auto& ptr2 = first_launch_ptrs[key.second];

                json pair_info;
                pair_info["param1_name"] = ptr1.name;
                pair_info["param1_index"] = ptr1.top_level_index;
                if (ptr1.member_index != -1) {
                    pair_info["param1_member_index"] = ptr1.member_index;
                }
                pair_info["param2_name"] = ptr2.name;
                pair_info["param2_index"] = ptr2.top_level_index;
                if (ptr2.member_index != -1) {
                    pair_info["param2_member_index"] = ptr2.member_index;
                }
                pair_info["noalias_ratio"] = ratio;
                noalias_pointers.push_back(pair_info);
            }
        }
    }

    if (!noalias_pointers.empty()) {
        hot_kernel_info["noalias_pointers"] = noalias_pointers;
    }
}

// --- Main Processing ---
void process_profile_data() {
    const char* json_path_env = std::getenv("CUDA_ARGS_PROFILE_JSON_FILE");
    if (!json_path_env) return;
    std::string json_path = json_path_env;

    std::ifstream read_file(json_path);
    if (!read_file.is_open()) {
        std::cerr << "Process Profile: Error opening file: " << json_path << std::endl;
        return;
    }

    json root;
    try {
        root = json::parse(read_file);
    } catch (json::parse_error& e) {
        std::cerr << "Process Profile: JSON parse error: " << e.what() << std::endl;
        return;
    }
    read_file.close();

    if (!root.contains("kernels") || !root["kernels"].is_array()) {
        return;
    }

    std::map<std::string, std::vector<json>> kernel_launches;
    for (const auto& launch : root["kernels"]) {
        kernel_launches[launch["name"]].push_back(launch);
    }

    std::vector<json> hot_kernels_vec;
    for (auto const& [name, launches] : kernel_launches) {
        if (launches.size() >= MIN_LAUNCH_COUNT) {
            json hot_kernel_info;
            hot_kernel_info["name"] = name;
            hot_kernel_info["launch_count"] = launches.size();
            find_noalias_pointers(launches, hot_kernel_info);
            find_common_scalars(launches, hot_kernel_info);
            hot_kernels_vec.push_back(hot_kernel_info);
        }
    }

    std::sort(hot_kernels_vec.begin(), hot_kernels_vec.end(), [](const json& a, const json& b) {
        return a["launch_count"].get<int>() > b["launch_count"].get<int>();
    });

    json new_root;
    new_root["hot_kernels"] = hot_kernels_vec;
    new_root["kernels"] = root["kernels"];

    std::ofstream write_file(json_path, std::ios::trunc);
    write_file << new_root.dump(2);
    write_file.close();
}
