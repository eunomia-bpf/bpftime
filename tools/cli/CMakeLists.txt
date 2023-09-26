add_executable(bpftime
  main.cpp
)

find_package(Boost REQUIRED COMPONENTS program_options)

add_dependencies(bpftime FridaCore bpftime-object)
target_include_directories(bpftime PUBLIC

  set(LIBBPF_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/../../runtime/object/libbpf)
  ${FRIDA_CORE_INSTALL_DIR}
  ../../runtime/object
  ../../third_party/
  ../../runtime/include
  ${LIBBPF_INCLUDE_DIRS}/uapi
  ${LIBBPF_INCLUDE_DIRS}
)

target_link_libraries(bpftime
  ${FRIDA_CORE_INSTALL_DIR}/libfrida-core.a
  -lpthread
  -lm
  -ldl
  runtime
  bpftime-object
  Boost::program_options
)

install(TARGETS bpftime CONFIGURATIONS Release Debug DESTINATION ~/.bpftime)
