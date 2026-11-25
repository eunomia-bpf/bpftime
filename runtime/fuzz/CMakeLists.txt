# libFuzzer targets for the runtime, built with -DBPFTIME_ENABLE_FUZZING=ON.
# The option instruments the whole tree with -fsanitize=fuzzer-no-link (see
# cmake/StandardSettings.cmake); only the fuzzer executables link the
# libFuzzer driver.
add_executable(bpftime_map_ops_fuzzer map_ops_fuzzer.cpp)
set_property(TARGET bpftime_map_ops_fuzzer PROPERTY CXX_STANDARD 20)
target_link_options(bpftime_map_ops_fuzzer PRIVATE -fsanitize=fuzzer)
target_include_directories(bpftime_map_ops_fuzzer PRIVATE ${BPFTIME_RUNTIME_INCLUDE})
target_link_libraries(bpftime_map_ops_fuzzer PRIVATE runtime)
add_dependencies(bpftime_map_ops_fuzzer runtime)
