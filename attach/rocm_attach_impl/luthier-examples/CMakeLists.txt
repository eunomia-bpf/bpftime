cmake_minimum_required(VERSION 3.23)

# The HIP fixture and tool DSO use the selected ROCm toolchain.
# The verification executable below is a normal target in bpftime's tree.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(bpftime_luthier_fixtures LANGUAGES CXX HIP)
  set(CMAKE_CXX_STANDARD 23)
  set(CMAKE_HIP_STANDARD 17)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  find_package(hip REQUIRED CONFIG)
  find_package(luthier REQUIRED CONFIG)

  add_library(bpftime_luthier_probe_tool SHARED probe_instrumentation.cpp)
  target_link_libraries(bpftime_luthier_probe_tool PRIVATE luthier::LuthierTooling)
  # Keep the private static LLVM instance separate from ROCm's shared LLVM.
  set_target_properties(bpftime_luthier_probe_tool PROPERTIES
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
  target_link_options(bpftime_luthier_probe_tool PRIVATE -Wl,--exclude-libs,ALL -Wl,-Bsymbolic)
  if(NOT LLVM_ENABLE_RTTI)
    target_compile_options(bpftime_luthier_probe_tool PRIVATE -fno-rtti)
  endif()
  add_executable(luthier_vector_add vector_add.hip)
  target_link_libraries(luthier_vector_add PRIVATE hip::host)
  target_compile_options(luthier_vector_add PRIVATE -O2 -g0)
  return()
endif()

set(_examples "${CMAKE_CURRENT_BINARY_DIR}/examples-build")

ExternalProject_Add(bpftime_luthier_fixtures
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" BINARY_DIR "${_examples}"
  DOWNLOAD_COMMAND "" UPDATE_COMMAND "" INSTALL_COMMAND ""
  DEPENDS bpftime_luthier_backend
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS ${BPFTIME_LUTHIER_CMAKE_ARGS} "-DCMAKE_MAKE_PROGRAM=${BPFTIME_LUTHIER_NINJA}"
    "-Dluthier_DIR=${BPFTIME_LUTHIER_BUILD_DIR}/lib/cmake/luthier"
  BUILD_COMMAND "${CMAKE_COMMAND}" --build "${_examples}"
    --parallel "${BPFTIME_LUTHIER_JOBS}"
  BUILD_BYPRODUCTS "${_examples}/luthier_vector_add"
    "${_examples}/${CMAKE_SHARED_LIBRARY_PREFIX}bpftime_luthier_probe_tool${CMAKE_SHARED_LIBRARY_SUFFIX}"
  BUILD_ALWAYS TRUE)

add_executable(bpftime_luthier_probe probe_tool.cpp)
target_compile_features(bpftime_luthier_probe PRIVATE cxx_std_17)
target_include_directories(bpftime_luthier_probe SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party")
target_compile_definitions(bpftime_luthier_probe PRIVATE
  BPFTIME_LUTHIER_APPLICATION="${_examples}/luthier_vector_add"
  BPFTIME_LUTHIER_TOOL_LIBRARY="${_examples}/${CMAKE_SHARED_LIBRARY_PREFIX}bpftime_luthier_probe_tool${CMAKE_SHARED_LIBRARY_SUFFIX}"
  BPFTIME_LUTHIER_RESULTS="${CMAKE_CURRENT_BINARY_DIR}/results")
if(BPFTIME_ENABLE_ROCM_ATTACH)
  target_compile_definitions(bpftime_luthier_probe PRIVATE
    BPFTIME_LUTHIER_AGENT_LIBRARY="$<TARGET_FILE:bpftime-agent>")
  add_dependencies(bpftime_luthier_probe bpftime-agent)
else()
  target_compile_definitions(bpftime_luthier_probe PRIVATE
    BPFTIME_LUTHIER_AGENT_LIBRARY="")
endif()
# Building the verifier always builds the independent HIP fixture and tool.
# No GPU work or verification runs until the executable is invoked.
add_dependencies(bpftime_luthier_probe bpftime_luthier_fixtures)
