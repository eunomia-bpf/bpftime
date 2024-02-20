add_library(
    bpftime_shm_common_library STATIC
    src/handler_manager.cpp

    src/handlers/epoll_handler.cpp
    src/handlers/map_handler.cpp
    src/handlers/perf_event_handler.cpp
    src/handlers/prog_handler.cpp

    src/maps/shared/array_map_kernel_user.cpp
    src/maps/shared/hash_map_kernel_user.cpp
    src/maps/shared/percpu_array_map_kernel_user.cpp
    src/maps/shared/perf_event_array_kernel_user.cpp

    src/maps/userspace/array_map.cpp
    src/maps/userspace/hash_map.cpp
    src/maps/userspace/per_cpu_array_map.cpp
    src/maps/userspace/per_cpu_hash_map.cpp
    src/maps/userspace/perf_event_array_map.cpp
    src/maps/userspace/prog_array.cpp
    src/maps/userspace/ringbuf_map.cpp
    
    
)

find_package(Boost REQUIRED)

set(SHM_COMMON_LIBRARY_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE STRING "Include path of shm common library")

add_dependencies(bpftime_shm_common_library spdlog::spdlog libbpf)

target_link_libraries(bpftime_shm_common_library PRIVATE spdlog::spdlog ${LIBBPF_LIBRARIES} z elf)
target_include_directories(bpftime_shm_common_library PRIVATE ${SHM_COMMON_LIBRARY_INCLUDE}   ${LIBBPF_INCLUDE_DIRS}/uapi
${LIBBPF_INCLUDE_DIRS})

set_property(TARGET bpftime_shm_common_library PROPERTY CXX_STANDARD 20)
