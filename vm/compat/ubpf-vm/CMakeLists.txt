add_library(bpftime_ubpf_vm_obj OBJECT ./compat_ubpf.cpp)

# add_subdirectory(ubpf)
set(UBPF_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/ubpf)
include(ExternalProject)
set(UBPF_CROSS_ARGS "")
if(CMAKE_CROSSCOMPILING)
    list(APPEND UBPF_CROSS_ARGS
        -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}
        -DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_SYSROOT=${CMAKE_SYSROOT}
    )
endif()
ExternalProject_Add(
    ubpf
    SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ubpf
    BINARY_DIR ${UBPF_BUILD_DIR}
    BUILD_BYPRODUCTS ${UBPF_BUILD_DIR}/lib/libubpf.a ${UBPF_BUILD_DIR}/vm/ubpf_config.h
    CMAKE_ARGS ${UBPF_CROSS_ARGS}
    INSTALL_COMMAND ""
)
add_dependencies(bpftime_ubpf_vm_obj ubpf spdlog::spdlog bpftime_vm_compat)

add_library(bpftime_ubpf_vm STATIC $<TARGET_OBJECTS:bpftime_ubpf_vm_obj>)
target_link_libraries(bpftime_ubpf_vm PUBLIC ${UBPF_BUILD_DIR}/lib/libubpf.a bpftime_vm_compat spdlog::spdlog)
ExternalProject_Get_Property(ubpf SOURCE_DIR)
target_include_directories(bpftime_ubpf_vm_obj PRIVATE ${SOURCE_DIR}/vm/inc ${UBPF_BUILD_DIR}/vm/ ${CMAKE_CURRENT_SOURCE_DIR} ${SPDLOG_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/../../vm-core/include ${CMAKE_CURRENT_SOURCE_DIR}/../include)
