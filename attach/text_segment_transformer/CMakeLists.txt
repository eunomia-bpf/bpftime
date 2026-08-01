add_library(bpftime_text_segment_transformer SHARED
    agent-transformer.cpp
    text_segment_transformer.cpp
)
add_dependencies(bpftime_text_segment_transformer spdlog::spdlog FridaGum)
set_target_properties(bpftime_text_segment_transformer PROPERTIES CXX_STANDARD 20 OUTPUT_NAME "bpftime-agent-transformer")

target_link_libraries(bpftime_text_segment_transformer
    spdlog::spdlog
    ${FRIDA_GUM_INSTALL_DIR}/libfrida-gum.a
)

target_include_directories(bpftime_text_segment_transformer
    PRIVATE
    ${FRIDA_GUM_INSTALL_DIR}
    ${SPDLOG_INCLUDE}
)

if (${TEST_LCOV}) 
    target_compile_options(bpftime_text_segment_transformer PRIVATE -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
    target_link_options(bpftime_text_segment_transformer PRIVATE -fprofile-arcs)
endif()
