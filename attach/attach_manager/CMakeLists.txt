add_library(
    bpftime_attach_manager STATIC
    src/instantiated_program.cpp
    src/attach_manager.cpp
)
set(ATTACH_MANAGER_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE STRING "Include path of attach manager")

add_dependencies(bpftime_attach_manager spdlog::spdlog vm-bpf)

target_link_libraries(bpftime_attach_manager PRIVATE spdlog::spdlog vm-bpf)
target_include_directories(bpftime_attach_manager PRIVATE ${SPDLOG_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include)

set_property(TARGET bpftime_attach_manager PROPERTY CXX_STANDARD 20)
