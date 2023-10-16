Include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.0.1
)

FetchContent_MakeAvailable(Catch2)

find_package(Boost REQUIRED)

set(TEST_SOURCES
    test_per_cpu_array.cpp
    test_per_cpu_hash.cpp
)

add_executable(bpftime_runtime_tests ${TEST_SOURCES})
set_property(TARGET bpftime_runtime_tests PROPERTY CXX_STANDARD 20)
add_dependencies(bpftime_runtime_tests runtime)
target_link_libraries(bpftime_runtime_tests PRIVATE runtime Catch2::Catch2WithMain)
target_include_directories(bpftime_runtime_tests PRIVATE ${BPFTIME_RUNTIME_INCLUDE} ${Catch2_INCLUDE} ${Boost_INCLUDE})
add_test(NAME bpftime_runtime_tests COMMAND bpftime_runtime_tests)
