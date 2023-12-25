add_executable(
    bpftime-vm-cli
    main.cpp
)

set_target_properties(bpftime-vm-cli PROPERTIES OUTPUT_NAME "bpftime-vm")


include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

target_include_directories(bpftime-vm-cli PRIVATE  ${SPDLOG_INCLUDE} ${argparse_INCLUDE})
target_link_libraries(bpftime-vm-cli PRIVATE spdlog::spdlog  argparse)
set_property(TARGET bpftime-vm-cli PROPERTY CXX_STANDARD 20)

target_compile_definitions(bpftime-vm-cli PRIVATE _GNU_SOURCE)

add_dependencies(bpftime-vm-cli spdlog::spdlog argparse)

install(TARGETS bpftime-vm-cli CONFIGURATIONS Release Debug DESTINATION ~/.bpftime)
