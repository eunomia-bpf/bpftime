
add_library(bpftime-object OBJECT
    bpf_object.cpp
  )

# frida-gum is optional: it is not built on architectures without frida
# support (riscv64). Without this guard FRIDA_GUM_INSTALL_DIR is empty and
# the link line degrades into the literal path "/libfrida-gum.a".
set(BPFTIME_OBJECT_FRIDA_LIBS "")
set(BPFTIME_OBJECT_FRIDA_INCLUDE "")
if(BPFTIME_ENABLE_FRIDA)
  set(BPFTIME_OBJECT_FRIDA_LIBS ${FRIDA_GUM_INSTALL_DIR}/libfrida-gum.a)
  set(BPFTIME_OBJECT_FRIDA_INCLUDE ${FRIDA_GUM_INSTALL_DIR})
endif()
if(${BPFTIME_BUILD_WITH_LIBBPF})
  add_dependencies(bpftime-object copy_headers libbpf spdlog::spdlog bpftime_vm)
  target_link_libraries(bpftime-object
    PUBLIC
    ${LIBBPF_LIBRARIES}
    ${BPFTIME_OBJECT_FRIDA_LIBS}
    -lpthread
    -lm
    -ldl
    -lelf
    -lz
    spdlog::spdlog
    bpftime_vm
  )

  target_include_directories(bpftime-object PUBLIC
    ${LIBBPF_INCLUDE_DIRS}/uapi
    ${LIBBPF_INCLUDE_DIRS}
    ${BPFTIME_OBJECT_FRIDA_INCLUDE}
    $<INSTALL_INTERFACE:runtime>
    $<INSTALL_INTERFACE:runtime/src>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../vm/include>
    ${SPDLOG_INCLUDE}
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} 
  )
else()
  add_dependencies(bpftime-object spdlog::spdlog bpftime_vm)
  target_link_libraries(bpftime-object
    PUBLIC
    ${BPFTIME_OBJECT_FRIDA_LIBS}
    -lpthread
    -lm
    -ldl
    -lelf
    -lz
    spdlog::spdlog
    bpftime_vm
  )

  target_include_directories(bpftime-object PUBLIC
    ${BPFTIME_OBJECT_FRIDA_INCLUDE}
    $<INSTALL_INTERFACE:runtime>
    $<INSTALL_INTERFACE:runtime/src>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../vm/include>
    ${SPDLOG_INCLUDE}
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} 
  )
endif()
set(BPFTIME_OBJECT_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
