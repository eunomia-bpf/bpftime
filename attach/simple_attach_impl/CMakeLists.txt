add_library(
    bpftime_simple_attach_impl STATIC
    ./simple_attach_impl.cpp
)

add_dependencies(bpftime_simple_attach_impl bpftime_base_attach_impl spdlog::spdlog)
set(SIMPLE_ATTACH_IMPL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "Include path of simple attach impl")

target_include_directories(bpftime_simple_attach_impl PRIVATE ${SPDLOG_INCLUDE} ${SIMPLE_ATTACH_IMPL_INCLUDE} PUBLIC ${BASE_ATTACH_IMPL_INCLUDE})

target_link_libraries(bpftime_simple_attach_impl PUBLIC bpftime_base_attach_impl spdlog::spdlog)
