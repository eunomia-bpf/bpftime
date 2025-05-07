# cmake version
cmake_minimum_required(VERSION 3.16.3)

# project info
project(pos_unittest LANGUAGES CXX CUDA)

# set executable output path
# execute_process( COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/../../build)
# SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../build)

# per-API test
add_executable(pos-cli ./main.cpp ./migrate.cpp)

# set(TEST_TARGETS per_api_test hidden_api_test)
set(TEST_TARGETS pos-cli)

foreach( test_target ${TEST_TARGETS} )
    set_target_properties(${test_target} PROPERTIES LINKER_LANGUAGE CUDA)

    # link libraries
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../lib)
    target_link_libraries(${test_target} -lcuda -lcublas -ldl -lpthread)
    find_library(pos_so NAMES pos HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../../lib)
    target_link_libraries(${test_target} ${pos_so})
    find_library(clang_so NAMES clang HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../../lib)
    target_link_libraries(${test_target} ${clang_so})

    # set include path
    target_include_directories(${test_target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)         # POS root path
    target_include_directories(${test_target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../lib)     # built headers

    # important here to set cuda as dynamically linked
    set_target_properties(${test_target} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    target_compile_features(${test_target} PUBLIC cxx_std_17)
    target_compile_options(
        ${test_target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL: -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80>
    )
endforeach( test_target ${TEST_TARGETS} )
