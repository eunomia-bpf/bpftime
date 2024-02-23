add_library(
    bpftime_shm_client STATIC
    src/handler_manager.cpp
)

set(SHM_CLIENT_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE STRING "Include path of shm client")

add_dependencies(bpftime_shm_client spdlog::spdlog bpftime_shm_common_library)

target_link_libraries(bpftime_shm_client PRIVATE spdlog::spdlog bpftime_shm_common_library)
target_include_directories(bpftime_shm_client PRIVATE ${SHM_CLIENT_INCLUDE} ${SHM_COMMON_LIBRARY_INCLUDE})

set_property(TARGET bpftime_shm_client PROPERTY CXX_STANDARD 20)
