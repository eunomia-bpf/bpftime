
add_library(
    attach_impl_example_nginx_plugin
    SHARED
    ./nginx_plugin/nginx_plugin.cpp
)

add_dependencies(attach_impl_example_nginx_plugin bpftime_base_attach_impl runtime spdlog::spdlog)

target_link_libraries(attach_impl_example_nginx_plugin PRIVATE bpftime_base_attach_impl runtime spdlog::spdlog)
target_include_directories(attach_impl_example_nginx_plugin PRIVATE ${BPFTIME_RUNTIME_INCLUDE} ${SPDLOG_INCLUDE})

add_executable(attach_impl_example_controller
    ./controller/controller.cpp
)

add_dependencies(attach_impl_example_controller runtime spdlog::spdlog)

target_link_libraries(attach_impl_example_controller PRIVATE runtime spdlog::spdlog)
target_include_directories(attach_impl_example_controller PRIVATE ${BPFTIME_RUNTIME_INCLUDE} ${SPDLOG_INCLUDE})
