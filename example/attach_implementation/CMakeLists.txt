add_library(
    attach_impl_example_nginx_plugin_adaptor
    SHARED
    ./nginx_plugin_adaptor/nginx_plugin_adaptor.cpp
)

add_dependencies(attach_impl_example_nginx_plugin_adaptor bpftime_simple_attach_impl runtime spdlog::spdlog bpftime_llvm_vm)

target_link_libraries(attach_impl_example_nginx_plugin_adaptor PRIVATE bpftime_simple_attach_impl runtime spdlog::spdlog bpftime_llvm_vm)
if(UNIX AND NOT APPLE)
  # Ensure VM factory registration (llvm) constructors are kept
  target_link_options(attach_impl_example_nginx_plugin_adaptor PUBLIC "-Wl,--whole-archive" "$<TARGET_FILE:bpftime_llvm_vm>" "-Wl,--no-whole-archive")
endif()
target_include_directories(attach_impl_example_nginx_plugin_adaptor PRIVATE ${BPFTIME_RUNTIME_INCLUDE} ${SPDLOG_INCLUDE} ${SIMPLE_ATTACH_IMPL_INCLUDE})

# add controller target
add_subdirectory(controller)

# We need to download the source of nginx, since it requires them to build modules
# Build nginx once with all modules included
ExternalProject_Add(
    nginx_build
    DOWNLOAD_COMMAND URL https://nginx.org/download/nginx-1.22.1.tar.gz
    CONFIGURE_COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/nginx_build-prefix/src/nginx_build && 
                     LINK_LIB=${CMAKE_CURRENT_BINARY_DIR}/libattach_impl_example_nginx_plugin_adaptor.so
                     ./configure 
                     --add-dynamic-module=${CMAKE_CURRENT_SOURCE_DIR}/nginx_plugin 
                     --add-dynamic-module=${CMAKE_CURRENT_SOURCE_DIR}/benchmark/baseline_nginx_plugin 
                     --add-dynamic-module=${CMAKE_CURRENT_SOURCE_DIR}/benchmark/dynamic_load_plugin 
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

# for benchmark
add_subdirectory(benchmark)

add_dependencies(nginx_build 
    attach_impl_example_nginx_plugin_adaptor 
    attach_impl_example_controller
)

# This is to ensure backward compatibility for any targets that depend on the old target name
add_custom_target(attach_impl_example_nginx DEPENDS nginx_build)
