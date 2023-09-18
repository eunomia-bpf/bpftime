add_library(bpftime-agent-transformer SHARED
    agent-transformer.cpp
    text_segment_transformer.cpp
)
set_property(TARGET bpftime-agent-transformer PROPERTY CXX_STANDARD 20)

target_link_libraries(bpftime-agent-transformer
    opcodes
)

target_include_directories(bpftime-agent-transformer
    PRIVATE
    ${FRIDA_GUM_INSTALL_DIR}
)
