#ifdef __cplusplus
extern "C" {
#endif

void __cuda_profile_kernel_launch(const char* kernel_name, int arg_count, void** arg_values, const char* arg_info_json_str);

#ifdef __cplusplus
}
#endif
