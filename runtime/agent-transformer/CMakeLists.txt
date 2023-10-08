option(USE_NEW_BINUTILS "Use the binutils version that requires four arguments in init_disassemble_info" NO)

add_library(bpftime-agent-transformer SHARED
    agent-transformer.cpp
    text_segment_transformer.cpp
)
add_dependencies(bpftime-agent-transformer spdlog::spdlog FridaGum)
set_property(TARGET bpftime-agent-transformer PROPERTY CXX_STANDARD 20)

target_link_libraries(bpftime-agent-transformer
    opcodes
    spdlog::spdlog
)

target_include_directories(bpftime-agent-transformer
    PRIVATE
    ${FRIDA_GUM_INSTALL_DIR}
    ${SPDLOG_INCLUDE}
)

if(${USE_NEW_BINUTILS})
    target_compile_definitions(bpftime-agent-transformer PRIVATE USE_NEW_BINUTILS)
endif()
