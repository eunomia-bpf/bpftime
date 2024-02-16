

add_library(bpftime_base_attach_impl STATIC 
    base_attach_impl.cpp
    attach_private_data.cpp
)

set(BASE_ATTACH_IMPL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "Include path of base_attach_impl")

add_dependencies(bpftime_base_attach_impl spdlog::spdlog)

target_link_libraries(bpftime_base_attach_impl PRIVATE spdlog::spdlog)
target_include_directories(bpftime_base_attach_impl PRIVATE ${SPDLOG_INCLUDE})

set_property(TARGET bpftime_base_attach_impl PROPERTY CXX_STANDARD 20)
