#
# Project settings
#

option(BPFTIME_BUILD_EXECUTABLE "Build the project as an executable, rather than a library." OFF)

#
# library options
#
option(BPFTIME_LLVM_JIT "Use LLVM as jit backend." OFF)

#
# Compiler options
#

option(BPFTIME_WARNINGS_AS_ERRORS "Treat compiler warnings as errors." OFF)

#
# Unit testing
#
# Currently supporting: GoogleTest, Catch2.

option(BPFTIME_ENABLE_UNIT_TESTING "Enable unit tests for the projects (from the `test` subfolder)." OFF)

option(BPFTIME_USE_CATCH2 "Use the Catch2 project for creating unit tests." OFF)

#
# Miscelanious options
#

# Generate compile_commands.json for clang based tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BPFTIME_VERBOSE_OUTPUT "Enable verbose output, allowing for a better understanding of each step taken." ON)

# Export all symbols when building a shared library
if(BUILD_SHARED_LIBS)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
  set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
endif()

option(BPFTIME_ENABLE_LTO "Enable Interprocedural Optimization, aka Link Time Optimization (LTO)." OFF)
if(BPFTIME_ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT result OUTPUT output)
  if(result)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(SEND_ERROR "IPO is not supported: ${output}.")
  endif()
endif()

option(BPFTIME_ENABLE_CCACHE "Enable the usage of Ccache, in order to speed up rebuild times." OFF)
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif()

option(BPFTIME_ENABLE_ASAN "Enable Address Sanitize to detect memory error." OFF)
if(BPFTIME_ENABLE_ASAN)
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
endif()

option(BPFTIME_ENABLE_MPK "Enable Memory Protection Keys for the share memory." OFF)
if(BPFTIME_ENABLE_MPK)
    add_definitions(-DBPFTIME_ENABLE_MPK)
endif()

option(BPFTIME_ENABLE_IOURING_EXT "Enable iouring helpers extensions." OFF)
if(BPFTIME_ENABLE_IOURING_EXT)
    add_definitions(-DBPFTIME_ENABLE_IOURING_EXT)
endif()

# whether to enable eBPF verifier in userspace
option(ENABLE_EBPF_VERIFIER "Whether to enable ebpf verifier" OFF)

# whether to build with bpftime daemon
option(BUILD_BPFTIME_DAEMON "Whether to build the bpftime daemon" ON)

# whether to build with shared bpf_map
option(BPFTIME_BUILD_KERNEL_BPF "Whether to build with bpf share maps" ON)

# whether to build single static library
option(BPFTIME_BUILD_STATIC_LIB "Whether to build a single static library for different archive files" OFF)

# whether to build bpftime with libbpf and other linux headers
option(BPFTIME_BUILD_WITH_LIBBPF "Whether to build with libbpf and other linux headers" ON)