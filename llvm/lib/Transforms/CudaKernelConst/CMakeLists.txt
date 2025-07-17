add_llvm_component_library(LLVMCudaKernelConst
  CudaKernelConst.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/CudaKernelConst

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  Core
  Support
  TargetParser
  TransformUtils
) 