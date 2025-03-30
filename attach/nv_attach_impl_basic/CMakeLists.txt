include(../../cmake/cuda.cmake)

add_library(
    bpftime_nv_attach_impl_basic STATIC
    nv_attach_impl_basic.cpp
)

add_dependencies(bpftime_nv_attach_impl_basic bpftime_base_attach_impl spdlog::spdlog)
set(NV_ATTACH_IMPL_BASIC_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ )
find_cuda()

target_include_directories(bpftime_nv_attach_impl_basic PRIVATE ${SPDLOG_INCLUDE} ${CMAKE_CURRENT_BINARY_DIR} PUBLIC ${CMAKE_SOURCE_DIR}/runtime/ ${BASE_ATTACH_IMPL_INCLUDE} ${NV_ATTACH_IMPL_BASIC_INCLUDE} ${CUDA_INCLUDE_PATH})

target_link_directories(bpftime_nv_attach_impl_basic PUBLIC ${CUDA_LIBRARY_PATH}
)

target_link_libraries(bpftime_nv_attach_impl_basic PUBLIC bpftime_base_attach_impl spdlog::spdlog ${CUDA_LIBS})

set_property(TARGET bpftime_nv_attach_impl_basic PROPERTY CXX_STANDARD 20)


if(${ENABLE_EBPF_VERIFIER} AND NOT TARGET Catch2)
    message(STATUS "Adding Catch2 by FetchContent at nv_attach_impl")
    Include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
    )
    FetchContent_MakeAvailable(Catch2)

    # if not enable verifier, we will use the catch2 from submodule
endif()
