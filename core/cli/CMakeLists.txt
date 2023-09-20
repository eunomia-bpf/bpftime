add_executable(bpftime-cli simple-cli.cpp)
set_property(TARGET bpftime-cli PROPERTY CXX_STANDARD 20)
add_dependencies(bpftime-cli core-bpf spdlog::spdlog)

target_include_directories(bpftime-cli PRIVATE ../include ${SPDLOG_INCLUDE})
target_link_libraries(bpftime-cli core-bpf spdlog::spdlog)
