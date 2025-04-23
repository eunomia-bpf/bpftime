cmake_minimum_required(VERSION 3.10)
project(nginx_dynamic_load_plugin)

# Nothing to build here - the nginx module is built by the parent CMakeLists.txt
# Filter implementation is built by its own Makefile

# Add a target to build the filter implementation
add_custom_target(
    filter_impl
    COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR}/libs
    COMMENT "Building filter implementation shared library"
)

# Make sure the filter_impl target is built before the nginx module
add_dependencies(filter_impl) 