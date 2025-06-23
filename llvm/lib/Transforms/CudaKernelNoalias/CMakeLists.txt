add_llvm_component_library(LLVMCudaKernelNoalias
  CudaKernelNoalias.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/CudaKernelNoalias

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  Core
  Support
  TargetParser
  TransformUtils
) 