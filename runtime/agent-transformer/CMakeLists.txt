add_library(bpftime-agent-transformer SHARED
    agent-transformer.cpp
    text_segment_transformer.cpp
)
add_dependencies(bpftime-agent-transformer spdlog::spdlog FridaGum)
set_property(TARGET bpftime-agent-transformer PROPERTY CXX_STANDARD 20)

target_link_libraries(bpftime-agent-transformer
    spdlog::spdlog
    ${FRIDA_GUM_INSTALL_DIR}/libfrida-gum.a
)

target_include_directories(bpftime-agent-transformer
    PRIVATE
    ${FRIDA_GUM_INSTALL_DIR}
    ${SPDLOG_INCLUDE}
)
