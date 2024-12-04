add_library(bpftime_preload SHARED
  main.cpp
)

target_include_directories(bpftime_preload PUBLIC
  ${LIBBPF_INCLUDE_DIRS}
  ../../third_party/
  ../../runtime/include
  ${LIBBPF_INCLUDE_DIRS}/uapi
  ${LIBBPF_INCLUDE_DIRS}
)

target_link_libraries(bpftime_preload
  -lpthread
  -lm
  -ldl
  runtime
  bpftime-object
)

# Set library properties
set_target_properties(bpftime_preload PROPERTIES
  PREFIX ""  # Remove 'lib' prefix
  OUTPUT_NAME "bpftime_preload"
)

install(TARGETS bpftime_preload 
  CONFIGURATIONS Release Debug RelWithDebInfo 
  LIBRARY DESTINATION ~/.bpftime
)
