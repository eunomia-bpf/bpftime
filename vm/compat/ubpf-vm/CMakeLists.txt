add_library(bpftime_ubpf_vm STATIC ./compat_ubpf.cpp)

# add_subdirectory(ubpf)
set(UBPF_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/ubpf)
include(ExternalProject)
ExternalProject_Add(
    ubpf
    SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ubpf
    BINARY_DIR ${UBPF_BUILD_DIR}
    BUILD_BYPRODUCTS ${UBPF_BUILD_DIR}/lib/libubpf.a ${UBPF_BUILD_DIR}/vm/ubpf_config.h
    # CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug
    INSTALL_COMMAND ""
)

add_dependencies(bpftime_ubpf_vm ubpf spdlog::spdlog bpftime_vm_compat)


target_link_libraries(bpftime_ubpf_vm PUBLIC ${UBPF_BUILD_DIR}/lib/libubpf.a bpftime_vm_compat spdlog::spdlog)
ExternalProject_Get_Property(ubpf SOURCE_DIR)
target_include_directories(bpftime_ubpf_vm PRIVATE ${SOURCE_DIR}/vm/inc ${UBPF_BUILD_DIR}/vm/ ${CMAKE_CURRENT_SOURCE_DIR} ${SPDLOG_INCLUDE})
