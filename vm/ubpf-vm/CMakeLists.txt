add_library(bpftime_ubpf_vm STATIC ./compat_ubpf.c)

# add_subdirectory(ubpf)
ExternalProject_Add(
    ubpf
    GIT_REPOSITORY https://github.com/iovisor/ubpf
    GIT_TAG 39872ad
)

add_dependencies(bpftime_ubpf_vm ubpf)

ExternalProject_Get_Property(ubpf BINARY_DIR)
target_link_libraries(bpftime_ubpf_vm PUBLIC ${BINARY_DIR}/lib/libubpf.a)
ExternalProject_Get_Property(ubpf SOURCE_DIR)
target_include_directories(bpftime_ubpf_vm PRIVATE ${SOURCE_DIR}/vm/inc ${CMAKE_CURRENT_SOURCE_DIR}/../compat/include)
