add_executable(
    bpftime-cli-cpp
    main.cpp
)

set_target_properties(bpftime-cli-cpp PROPERTIES OUTPUT_NAME "bpftime")

target_include_directories(bpftime-cli-cpp PRIVATE ${FRIDA_CORE_INSTALL_DIR} ${SPDLOG_INCLUDE} ${argparse_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/include ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/src ${CMAKE_CURRENT_SOURCE_DIR}/../../vm/vm-core/include)
target_link_libraries(bpftime-cli-cpp PRIVATE spdlog::spdlog ${FRIDA_CORE_INSTALL_DIR}/libfrida-core.a argparse spdlog::spdlog bpftime_vm_compat runtime)
set_property(TARGET bpftime-cli-cpp PROPERTY CXX_STANDARD 20)

target_compile_definitions(bpftime-cli-cpp PRIVATE _GNU_SOURCE)

add_dependencies(bpftime-cli-cpp spdlog::spdlog FridaCore argparse spdlog runtime bpftime_vm bpftime_vm_compat)

install(TARGETS bpftime-cli-cpp CONFIGURATIONS Release Debug RelWithDebInfo DESTINATION ~/.bpftime)
