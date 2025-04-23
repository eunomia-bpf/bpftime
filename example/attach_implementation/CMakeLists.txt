add_library(
    attach_impl_example_nginx_plugin_adaptor
    SHARED
    ./nginx_plugin_adaptor/nginx_plugin_adaptor.cpp
)

add_dependencies(attach_impl_example_nginx_plugin_adaptor bpftime_base_attach_impl runtime spdlog::spdlog)

target_link_libraries(attach_impl_example_nginx_plugin_adaptor PRIVATE bpftime_base_attach_impl runtime spdlog::spdlog)
target_include_directories(attach_impl_example_nginx_plugin_adaptor PRIVATE ${BPFTIME_RUNTIME_INCLUDE} ${SPDLOG_INCLUDE})

# add eBPF program target and controller executable target
add_executable(attach_impl_example_controller
    ./controller/controller.cpp
)
add_ebpf_program_target(
    attach_impl_example_ebpf_program
    ${CMAKE_CURRENT_SOURCE_DIR}/controller/request_filter.bpf.c
    ${CMAKE_CURRENT_BINARY_DIR}/request_filter.bpf.o
)
target_compile_definitions(attach_impl_example_controller PRIVATE EBPF_PROGRAM_PATH=${CMAKE_CURRENT_BINARY_DIR}/request_filter.bpf.o)

add_dependencies(attach_impl_example_controller runtime spdlog::spdlog bpftime-object attach_impl_example_ebpf_program)

target_link_libraries(attach_impl_example_controller PRIVATE runtime spdlog::spdlog bpftime-object)
target_include_directories(attach_impl_example_controller PRIVATE ${BPFTIME_RUNTIME_INCLUDE} ${SPDLOG_INCLUDE} ${BPFTIME_OBJECT_INCLUDE_DIRS})

# We need to download the source of nginx, since it requires them to build modules
# Build nginx once with both modules included
ExternalProject_Add(
    nginx_build
    DOWNLOAD_COMMAND URL https://nginx.org/download/nginx-1.22.1.tar.gz
    CONFIGURE_COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/nginx_build-prefix/src/nginx_build && 
                     LINK_LIB=${CMAKE_CURRENT_BINARY_DIR}/libattach_impl_example_nginx_plugin_adaptor.so
                     ./configure 
                     --add-dynamic-module=${CMAKE_CURRENT_SOURCE_DIR}/nginx_plugin 
                     --add-dynamic-module=${CMAKE_CURRENT_SOURCE_DIR}/benchmark/baseline_nginx_plugin 
                     --with-compat 
                     --with-ld-opt=-Wl,-rpath=${CMAKE_CURRENT_BINARY_DIR}
    BUILD_COMMAND make -C ${CMAKE_CURRENT_BINARY_DIR}/nginx_build-prefix/src/nginx_build -j32
    INSTALL_COMMAND ""
    BUILD_ALWAYS TRUE
    DEPENDS attach_impl_example_nginx_plugin_adaptor
)

add_custom_command(
    TARGET nginx_build
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/nginx_plugin_output
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/nginx_build-prefix/src/nginx_build/objs/*.so ${CMAKE_CURRENT_SOURCE_DIR}/nginx_plugin_output/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/nginx_build-prefix/src/nginx_build/objs/nginx ${CMAKE_CURRENT_SOURCE_DIR}/nginx_plugin_output/
    COMMENT "Copy nginx modules and executable to ${CMAKE_CURRENT_SOURCE_DIR}/nginx_plugin_output/"
)

add_dependencies(nginx_build attach_impl_example_nginx_plugin_adaptor attach_impl_example_controller)

# This is to ensure backward compatibility for any targets that depend on the old target name
add_custom_target(attach_impl_example_nginx DEPENDS nginx_build)
