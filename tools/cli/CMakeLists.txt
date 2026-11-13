add_executable(
    bpftime-cli-cpp
    main.cpp
)

set_target_properties(bpftime-cli-cpp PROPERTIES OUTPUT_NAME "bpftime")

if(APPLE)
  # Suppress Boost template warning on macOS (Boost/compiler compatibility issue)
  target_compile_options(bpftime-cli-cpp PRIVATE -Wno-missing-template-arg-list-after-template-kw)
  # Fix linker options for macOS
  if(${BPFTIME_UBPF_JIT})
    target_link_options(bpftime-cli-cpp PUBLIC "-Wl,-force_load,$<TARGET_FILE:bpftime_ubpf_vm>")
  endif()
  if(${BPFTIME_LLVM_JIT})
    target_link_options(bpftime-cli-cpp PUBLIC "-Wl,-force_load,$<TARGET_FILE:bpftime_llvm_vm>")
  endif()
else()
  # Linux linker options
  if(${BPFTIME_UBPF_JIT})
    target_link_options(bpftime-cli-cpp PUBLIC "-Wl,--whole-archive" "$<TARGET_FILE:bpftime_ubpf_vm>" "-Wl,--no-whole-archive")
  endif()
  if(${BPFTIME_LLVM_JIT})
    target_link_options(bpftime-cli-cpp PUBLIC "-Wl,--whole-archive" "$<TARGET_FILE:bpftime_llvm_vm>" "-Wl,--no-whole-archive")
  endif()
endif()

target_include_directories(bpftime-cli-cpp PRIVATE ${FRIDA_CORE_INSTALL_DIR} ${SPDLOG_INCLUDE} ${argparse_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/include ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/src ${CMAKE_CURRENT_SOURCE_DIR}/../../vm/vm-core/include)
target_link_libraries(bpftime-cli-cpp PRIVATE spdlog::spdlog ${FRIDA_CORE_INSTALL_DIR}/libfrida-core.a argparse spdlog::spdlog bpftime_vm_compat runtime)
# Link the VM compat static libs exactly once via runtime to avoid duplicate symbols
# Linker options are handled conditionally above for macOS vs Linux
if(${BPFTIME_UBPF_JIT})
  add_dependencies(bpftime-cli-cpp bpftime_ubpf_vm)
  target_link_libraries(bpftime-cli-cpp PUBLIC bpftime_ubpf_vm)
endif()
if(${BPFTIME_LLVM_JIT})
  add_dependencies(bpftime-cli-cpp bpftime_llvm_vm llvmbpf_vm)
  target_link_libraries(bpftime-cli-cpp PUBLIC bpftime_llvm_vm)
endif()
set_property(TARGET bpftime-cli-cpp PROPERTY CXX_STANDARD 20)

target_compile_definitions(bpftime-cli-cpp PRIVATE _GNU_SOURCE)

add_dependencies(bpftime-cli-cpp spdlog::spdlog FridaCore argparse spdlog runtime bpftime_vm bpftime_vm_compat)

install(TARGETS bpftime-cli-cpp CONFIGURATIONS Release Debug RelWithDebInfo DESTINATION ~/.bpftime)
