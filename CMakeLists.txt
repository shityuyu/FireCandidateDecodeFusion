cmake_minimum_required(VERSION 3.16)
project(FireCandidateDecodeFusion LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# 环境检查：必须设置 ASCEND_HOME_PATH，例如 /usr/local/Ascend
# ---------------------------------------------------------------------------
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH is not set. export ASCEND_HOME_PATH=/usr/local/Ascend")
endif()
set(ASCEND_HOME $ENV{ASCEND_HOME_PATH})

# CANN toolkit 根目录（兼容不同版本目录布局）
if(EXISTS "${ASCEND_HOME}/ascend-toolkit/latest")
    set(CANN_ROOT "${ASCEND_HOME}/ascend-toolkit/latest")
elseif(EXISTS "${ASCEND_HOME}/latest")
    set(CANN_ROOT "${ASCEND_HOME}/latest")
else()
    set(CANN_ROOT "${ASCEND_HOME}")
endif()

# ---------------------------------------------------------------------------
# 头文件搜索路径
# ---------------------------------------------------------------------------
include_directories(
    ${CANN_ROOT}/include
    ${CANN_ROOT}/include/aclnn
    ${CANN_ROOT}/include/opapi
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

# ---------------------------------------------------------------------------
# kernel：使用 CANN 提供的 ccec(bisheng) 编译器编译，产物为静态库
# 编译器路径可显式通过 -DKERNEL_CXX=... 覆盖
# ---------------------------------------------------------------------------
if(NOT DEFINED KERNEL_CXX)
    set(KERNEL_CXX "${CANN_ROOT}/compiler/ccec_compiler/bin/clang++")
endif()
message(STATUS "KERNEL_CXX = ${KERNEL_CXX}")

add_library(fire_candidate_decode_fusion_kernel STATIC
    op_kernel/fire_candidate_decode_fusion.cpp
)
set_target_properties(fire_candidate_decode_fusion_kernel PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
# kernel 源文件使用 ccec 编译器
if(EXISTS "${KERNEL_CXX}")
    set_source_files_properties(op_kernel/fire_candidate_decode_fusion.cpp PROPERTIES
        COMPILE_FLAGS "--cce-aicore-arch=..."
    )
endif()

# ---------------------------------------------------------------------------
# host：使用系统 g++ 编译（tiling + op 注册 + aclnn 接口）
# ---------------------------------------------------------------------------
add_library(fire_candidate_decode_fusion_host STATIC
    op_host/fire_candidate_decode_fusion.cpp
    op_host/fire_candidate_decode_fusion_tiling.cpp
    fire_candidate_decode_fusion.cpp
)

# ---------------------------------------------------------------------------
# 合并为最终动态库 libfire_candidate_decode_fusion.so
# ---------------------------------------------------------------------------
add_library(fire_candidate_decode_fusion SHARED $<TARGET_OBJECTS:fire_candidate_decode_fusion_kernel>
    $<TARGET_OBJECTS:fire_candidate_decode_fusion_host>)
