set(GPU_SOURCES
    src/gpu/gpu_platform.cpp
    src/gpu/uniformity_analysis.cpp
    src/gpu/resource_budget.cpp
    src/gpu/simt_safety_check.cpp
    src/gpu/gpu_verifier.cpp
)

add_library(bpftime-verifier STATIC
    src/bpftime-verifier.cpp
    src/platform-impl.cpp
    ${GPU_SOURCES}
)
add_subdirectory(ebpf-verifier)

set_property(TARGET bpftime-verifier PROPERTY CXX_STANDARD 20)

target_include_directories(bpftime-verifier
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ebpf-verifier/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu
    ${CMAKE_CURRENT_SOURCE_DIR}/../vm/compat/include
    ${LIBBPF_INCLUDE_DIRS}
)

target_link_libraries(bpftime-verifier PUBLIC ebpfverifier ${LIBBPF_LIBRARIES} z elf)

add_dependencies(bpftime-verifier ebpfverifier libbpf)

set(BPFTIME_VERIFIER_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(TEST_SOURCES
    test/simple.cpp
    test/map.cpp
    test/non_kernel_helper.cpp
)

find_program(BPFTIME_CLANG_EXECUTABLE NAMES clang REQUIRED)

set(BPFTIME_GPU_BPF_OBJECT_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/generated-gpu-bpf
)
set(BPFTIME_GPU_BPF_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/vmlinux
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/bpftool/libbpf/src
)
set(BPFTIME_GPU_BPF_ARCH "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    set(BPFTIME_GPU_BPF_ARCH x86)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(BPFTIME_GPU_BPF_ARCH arm64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
    set(BPFTIME_GPU_BPF_ARCH arm)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv64")
    set(BPFTIME_GPU_BPF_ARCH riscv)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^ppc64")
    set(BPFTIME_GPU_BPF_ARCH powerpc)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(loongarch64|loong64)$")
    set(BPFTIME_GPU_BPF_ARCH loongarch)
endif()
set(BPFTIME_GPU_BPF_INCLUDE_FLAGS)
foreach(BPFTIME_GPU_BPF_INCLUDE_DIR ${BPFTIME_GPU_BPF_INCLUDE_DIRS})
    list(APPEND BPFTIME_GPU_BPF_INCLUDE_FLAGS
        -I${BPFTIME_GPU_BPF_INCLUDE_DIR}
    )
endforeach()
set(BPFTIME_GPU_BPF_COMPILE_FLAGS
    -target bpf
    -O2
    -g
    -c
)
if(BPFTIME_GPU_BPF_ARCH)
    list(APPEND BPFTIME_GPU_BPF_COMPILE_FLAGS
        -D__TARGET_ARCH_${BPFTIME_GPU_BPF_ARCH}
    )
endif()
set(BPFTIME_GPU_BPF_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/test/gpu_unsafe_programs/varying_branch.bpf.c
    ${CMAKE_CURRENT_SOURCE_DIR}/test/gpu_unsafe_programs/prohibited_helper.bpf.c
    ${CMAKE_CURRENT_SOURCE_DIR}/test/gpu_unsafe_programs/varying_atomic.bpf.c
    ${CMAKE_CURRENT_SOURCE_DIR}/test/gpu_unsafe_programs/varying_map_key.bpf.c
    ${CMAKE_CURRENT_SOURCE_DIR}/test/gpu_unsafe_programs/resource_exceeded.bpf.c
    ${CMAKE_CURRENT_SOURCE_DIR}/test/gpu_unsafe_programs/safe_counter.bpf.c
    ${CMAKE_CURRENT_SOURCE_DIR}/test/gpu_unsafe_programs/safe_block_idx_branch.bpf.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../example/gpu/cuda-counter/cuda_probe.bpf.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../example/gpu/gpu_shared_map/gpu_shared_map.bpf.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../example/gpu/rocm-counter/rocm_probe.bpf.c
)

set(BPFTIME_GPU_BPF_OBJECTS)
foreach(BPF_SOURCE ${BPFTIME_GPU_BPF_SOURCES})
    get_filename_component(BPF_SOURCE_NAME ${BPF_SOURCE} NAME)
    string(REPLACE ".c" ".o" BPF_OBJECT_NAME ${BPF_SOURCE_NAME})
    set(BPF_OBJECT_PATH ${BPFTIME_GPU_BPF_OBJECT_DIR}/${BPF_OBJECT_NAME})

    add_custom_command(
        OUTPUT ${BPF_OBJECT_PATH}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${BPFTIME_GPU_BPF_OBJECT_DIR}
        COMMAND ${BPFTIME_CLANG_EXECUTABLE}
                ${BPFTIME_GPU_BPF_COMPILE_FLAGS}
                ${BPF_SOURCE}
                -o ${BPF_OBJECT_PATH}
                ${BPFTIME_GPU_BPF_INCLUDE_FLAGS}
        DEPENDS ${BPF_SOURCE}
        COMMAND_EXPAND_LISTS
        VERBATIM
    )

    list(APPEND BPFTIME_GPU_BPF_OBJECTS ${BPF_OBJECT_PATH})
endforeach()

add_custom_target(bpftime_gpu_bpf_objects DEPENDS ${BPFTIME_GPU_BPF_OBJECTS})

if(NOT TARGET Catch2)
    set(BPFTIME_CATCH2_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/Catch2)
    if(EXISTS ${BPFTIME_CATCH2_DIR}/CMakeLists.txt)
        message(STATUS "Adding Catch2 by subdirectory at bpftime-verifier")
        add_subdirectory(${BPFTIME_CATCH2_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/third_party/Catch2)
    else()
        message(STATUS "Adding Catch2 by FetchContent at bpftime-verifier")
        Include(FetchContent)

        FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v3.4.0
        )

        FetchContent_MakeAvailable(Catch2)
    endif()
endif()

option(TEST_LCOV "option for lcov" OFF)
add_executable(bpftime_verifier_tests ${TEST_SOURCES})
add_executable(bpftime_gpu_verifier_tests test/gpu_verifier_test.cpp)
add_executable(bpftime_gpu_verifier_e2e_tests test/gpu_verifier_e2e_test.cpp)
add_executable(bpftime_gpu_verifier_stress_tests test/gpu_verifier_stress_test.cpp)
add_executable(bpftime_gpu_verify_perf test/gpu_verify_perf.cpp)
add_executable(bpftime_gpu_verifier_eval test/gpu_verifier_eval.cpp)

if (${TEST_LCOV}) 
    target_compile_options(bpftime_verifier_tests PRIVATE -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
    target_compile_options(bpftime_gpu_verifier_tests PRIVATE -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
    target_compile_options(bpftime_gpu_verifier_e2e_tests PRIVATE -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
    target_compile_options(bpftime_gpu_verifier_stress_tests PRIVATE -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
endif()

add_dependencies(bpftime_verifier_tests bpftime-verifier)
add_dependencies(bpftime_gpu_verifier_tests bpftime-verifier)
add_dependencies(bpftime_gpu_verifier_e2e_tests bpftime-verifier bpftime_gpu_bpf_objects)
add_dependencies(bpftime_gpu_verifier_stress_tests bpftime-verifier)
add_dependencies(bpftime_gpu_verify_perf bpftime-verifier)
add_dependencies(bpftime_gpu_verifier_eval bpftime-verifier)

if (${TEST_LCOV}) 
    target_link_options(bpftime_verifier_tests PRIVATE -lgcov)
    target_link_libraries(bpftime_verifier_tests PRIVATE bpftime-verifier Catch2::Catch2WithMain gcov)
    target_link_options(bpftime_gpu_verifier_tests PRIVATE -lgcov)
    target_link_libraries(bpftime_gpu_verifier_tests PRIVATE bpftime-verifier Catch2::Catch2WithMain gcov)
    target_link_options(bpftime_gpu_verifier_e2e_tests PRIVATE -lgcov)
    target_link_libraries(bpftime_gpu_verifier_e2e_tests PRIVATE bpftime-verifier Catch2::Catch2WithMain gcov)
    target_link_options(bpftime_gpu_verifier_stress_tests PRIVATE -lgcov)
    target_link_libraries(bpftime_gpu_verifier_stress_tests PRIVATE bpftime-verifier Catch2::Catch2WithMain gcov)
else ()
    target_link_libraries(bpftime_verifier_tests PRIVATE bpftime-verifier Catch2::Catch2WithMain)
    target_link_libraries(bpftime_gpu_verifier_tests PRIVATE bpftime-verifier Catch2::Catch2WithMain)
    target_link_libraries(bpftime_gpu_verifier_e2e_tests PRIVATE bpftime-verifier Catch2::Catch2WithMain)
    target_link_libraries(bpftime_gpu_verifier_stress_tests PRIVATE bpftime-verifier Catch2::Catch2WithMain)
endif()
target_link_libraries(bpftime_gpu_verify_perf PRIVATE bpftime-verifier)
target_link_libraries(bpftime_gpu_verifier_eval PRIVATE bpftime-verifier)
target_include_directories(bpftime_verifier_tests PRIVATE ${BPFTIME_VERIFIER_INCLUDE})
target_include_directories(bpftime_gpu_verifier_tests PRIVATE ${BPFTIME_VERIFIER_INCLUDE})
target_include_directories(bpftime_gpu_verifier_e2e_tests PRIVATE ${BPFTIME_VERIFIER_INCLUDE})
target_include_directories(bpftime_gpu_verifier_stress_tests PRIVATE ${BPFTIME_VERIFIER_INCLUDE})
target_include_directories(bpftime_gpu_verifier_eval PRIVATE ${BPFTIME_VERIFIER_INCLUDE})
target_compile_definitions(bpftime_gpu_verifier_e2e_tests PRIVATE
    BPFTIME_GPU_BPF_OBJECT_DIR=\"${BPFTIME_GPU_BPF_OBJECT_DIR}\"
)
add_test(NAME bpftime_verifier_tests COMMAND bpftime_verifier_tests)
add_test(NAME bpftime_gpu_verifier_tests COMMAND bpftime_gpu_verifier_tests)
add_test(NAME bpftime_gpu_verifier_e2e_tests COMMAND bpftime_gpu_verifier_e2e_tests)
add_test(NAME bpftime_gpu_verifier_stress_tests COMMAND bpftime_gpu_verifier_stress_tests)
set_target_properties(bpftime_gpu_verify_perf
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bpftime-verifier
)
set_target_properties(bpftime_gpu_verifier_e2e_tests
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bpftime-verifier
)
set_target_properties(bpftime_gpu_verifier_eval
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bpftime-verifier
)
