add_library(bpftime-agent SHARED 
  agent.cpp
)
add_dependencies(bpftime-agent FridaGum)
target_include_directories(bpftime-agent 
    PRIVATE 
    ${FRIDA_GUM_INSTALL_DIR}
    ../include
    ../../third_party/
)
target_link_libraries(bpftime-agent 
    ${FRIDA_GUM_INSTALL_DIR}/libfrida-gum.a
    runtime
    -lpthread
    -lm 
    -ldl
)
