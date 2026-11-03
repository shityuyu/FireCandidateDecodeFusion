# Kernel sources are collected from op_kernel.
npu_op_kernel_sources(ascendc_kernels
    KERNEL_DIR ./
)

# Kernel library linked with the Host tiling library.
npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${package_name}
    LIBRARY ascendc_kernels
)
