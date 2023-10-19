
add_library(bpftime-object OBJECT
  bpf_object.cpp
)

add_dependencies(bpftime-object copy_headers)
add_dependencies(bpftime-object libbpf)
add_dependencies(bpftime-object spdlog::spdlog)

target_link_libraries(bpftime-object
  PUBLIC
  ${LIBBPF_LIBRARIES}
  ${FRIDA_GUM_INSTALL_DIR}/libfrida-gum.a
  -lpthread
  -lm
  -ldl
  -lelf
  -lz
  spdlog::spdlog
)

target_include_directories(bpftime-object PUBLIC
  ${LIBBPF_INCLUDE_DIRS}/uapi
  ${LIBBPF_INCLUDE_DIRS}
  ${FRIDA_GUM_INSTALL_DIR}
  $<INSTALL_INTERFACE:runtime>
  $<INSTALL_INTERFACE:runtime/src>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../vm/include>
  ${SPDLOG_INCLUDE}
)
set(BPFTIME_OBJECT_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
