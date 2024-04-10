# Find all headers and implementation files
# Allow user to specify the architecture
if(NOT DEFINED ARCH)
  set(ARCH ${CMAKE_SYSTEM_PROCESSOR})
endif()

message(STATUS "Building for architecture: ${ARCH}")

# Detect the architecture
if(ARCH MATCHES "arm")
  message(STATUS "arm architecture detected")
  # set(ARCH_HEADERS
  #   arch/arm/
  # )
elseif(ARCH MATCHES "aarch64")
  message(STATUS "arm64 architecture detected")
  set(ARCH_SOURCES
    ebpf_jit_arm64.c
  )
  # set(ARCH_HEADERS
  #   arch/arm64/
  # )
elseif(ARCH MATCHES "x86_64" OR ARCH MATCHES "i686" OR ARCH MATCHES "i386")
  message(STATUS "x86 architecture detected")
  set(ARCH_SOURCES
    ebpf_jit_x86_64.c
  )
  # set(ARCH_HEADERS
  #   arch/x86/
  # )
else()
  message(FATAL_ERROR "Unsupported architecture")
endif()

set(sources
  ${ARCH_SOURCES}
  ebpf_jit.c
  ebpf_vm.c
)

set(exe_sources
  ../example/main.c
  ${sources}
)

set(headers
  include/
  vm/
  # ${ARCH_HEADERS}
  ${headerfiles}
)
message(STATUS ${headers})

#
# Create library, setup header and source files
#
add_library(
  bpftime_simple_jit
  ${headers}
  ${sources}
)
set_target_properties(bpftime_simple_jit PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/.." ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/../")
add_executable(bpftime_simple_jit-cli ${exe_sources})
target_link_libraries(bpftime_simple_jit-cli PUBLIC bpftime_simple_jit)

message(DEBUG "Found the following sources:")

foreach(source IN LISTS exe_sources)
  message(DEBUG "* ${source}")
endforeach()

target_include_directories(
  bpftime_simple_jit
  PUBLIC
  ${headers}
  ./
  ${CMAKE_CURRENT_SOURCE_DIR}/../compat/include
)

target_include_directories(bpftime_simple_jit PUBLIC ../include)
