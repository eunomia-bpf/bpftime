cmake_minimum_required(VERSION 3.23)

# The HIP fixture and tool DSO need ROCm's compiler and private LLVM instance.
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

# Luthier uses ROCm's LLVM internals; bpftime may use a different host LLVM.
include(ExternalProject)
find_program(BPFTIME_LUTHIER_NINJA ninja REQUIRED)

set(BPFTIME_LUTHIER_LLVM_SOURCE "" CACHE PATH "Optional matching llvm-project source (contains llvm/); empty downloads it at build time")
set(BPFTIME_LUTHIER_ARCH "" CACHE STRING "AMDGPU architecture for Luthier, e.g. gfx1201")
set(BPFTIME_LUTHIER_JOBS 8 CACHE STRING "Parallel jobs for the isolated Luthier build")
if(NOT BPFTIME_ROCM_ROOT OR NOT BPFTIME_LUTHIER_ARCH)
  message(FATAL_ERROR "Set BPFTIME_ROCM_ROOT and BPFTIME_LUTHIER_ARCH; see luthier-examples/README.md")
endif()
set(_llvm "${BPFTIME_ROCM_ROOT}/lib/llvm")
execute_process(COMMAND "${_llvm}/bin/llvm-config" --version
  OUTPUT_VARIABLE _llvm_version OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _llvm_status)
if(NOT _llvm_status EQUAL 0 OR NOT _llvm_version MATCHES "^24\\.")
  message(FATAL_ERROR "The Luthier fork is validated with package LLVM 24")
endif()
execute_process(COMMAND "${_llvm}/bin/llvm-config" --has-rtti
  OUTPUT_VARIABLE _llvm_rtti OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
set(_rtti_flags "")
if(_llvm_rtti STREQUAL "NO")
  set(_rtti_flags "-fno-rtti")
endif()

set(_llvm_source_dependency "")
if(BPFTIME_LUTHIER_LLVM_SOURCE)
  set(_llvm_source "${BPFTIME_LUTHIER_LLVM_SOURCE}")
  foreach(_file IN ITEMS llvm/lib/Target/AMDGPU/AMDGPU.td
      llvm/include/llvm/IR/Intrinsics.td llvm/utils/TableGen/Common/CodeGenTarget.h)
    if(NOT EXISTS "${_llvm_source}/${_file}")
      message(FATAL_ERROR "BPFTIME_LUTHIER_LLVM_SOURCE is missing ${_file}; point it to matching LLVM sources or leave it empty to download automatically")
    endif()
  endforeach()
else()
  # Luthier consumes private LLVM APIs, so fetch the installed build's exact
  # revision rather than assuming that any source with the same major matches.
  set(_vcs_header "${_llvm}/include/llvm/Support/VCSRevision.h")
  if(NOT EXISTS "${_vcs_header}")
    message(FATAL_ERROR "Cannot detect LLVM source revision: ${_vcs_header} is missing. Set BPFTIME_LUTHIER_LLVM_SOURCE manually")
  endif()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_vcs_header}")
  file(READ "${_vcs_header}" _vcs)
  # LLVM uses either ordinary string literals or raw R\"(...)\" literals.
  string(REGEX MATCH "#define LLVM_REVISION[ \t]+(R\"\\(|\")([0-9a-fA-F]+)" _match "${_vcs}")
  set(_revision "${CMAKE_MATCH_2}")
  string(LENGTH "${_revision}" _revision_length)
  string(REGEX MATCH "#define LLVM_REPOSITORY[ \t]+(R\"\\(|\")https://github.com/([A-Za-z0-9_.-]+)/llvm-project(\\.git)?" _match "${_vcs}")
  set(_owner "${CMAKE_MATCH_2}")
  if(NOT _revision_length EQUAL 40 OR NOT _owner)
    message(FATAL_ERROR "Cannot detect a GitHub llvm-project repository and full commit from ${_vcs_header}. Set BPFTIME_LUTHIER_LLVM_SOURCE manually")
  endif()
  set(_llvm_cache "${CMAKE_BINARY_DIR}/_deps/luthier-llvm-${_owner}-${_revision}")
  set(_llvm_source "${_llvm_cache}/src/llvm-project-${_revision}")
  message(STATUS "Luthier LLVM sources: ${_owner}/llvm-project@${_revision} (downloaded at build time to ${_llvm_source})")
  ExternalProject_Add(bpftime_luthier_llvm_source
    PREFIX "${_llvm_cache}"
    URL "https://codeload.github.com/${_owner}/llvm-project/tar.gz/${_revision}"
    DOWNLOAD_DIR "${_llvm_cache}/download"
    DOWNLOAD_NAME "${_revision}.tar.gz"
    DOWNLOAD_NO_EXTRACT TRUE
    TLS_VERIFY TRUE
    SOURCE_DIR "${_llvm_cache}/src"
    CONFIGURE_COMMAND "${CMAKE_COMMAND}"
      "-DARCHIVE=<DOWNLOADED_FILE>" "-DDESTINATION=<SOURCE_DIR>"
      "-DREVISION=${_revision}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/extract_llvm_source.cmake"
    UPDATE_COMMAND "" BUILD_COMMAND "" INSTALL_COMMAND "")
  ExternalProject_Add_StepDependencies(bpftime_luthier_llvm_source configure
    "${CMAKE_CURRENT_SOURCE_DIR}/extract_llvm_source.cmake")
  set(_llvm_source_dependency bpftime_luthier_llvm_source)
endif()

set(_source "${CMAKE_CURRENT_SOURCE_DIR}/../luthier")
set(_backend "${CMAKE_CURRENT_BINARY_DIR}/luthier-build")
set(_examples "${CMAKE_CURRENT_BINARY_DIR}/examples-build")
if(NOT EXISTS "${_source}/CMakeLists.txt")
  message(FATAL_ERROR "Initialize the attach/rocm_attach_impl/luthier submodule")
endif()
set(_common_args
  "-DCMAKE_CXX_COMPILER=${_llvm}/bin/clang++"
  "-DCMAKE_HIP_COMPILER=${_llvm}/bin/clang++"
  "-DCMAKE_HIP_COMPILER_ROCM_ROOT=${BPFTIME_ROCM_ROOT}"
  "-DCMAKE_PREFIX_PATH=${BPFTIME_ROCM_ROOT}"
  "-DLLVM_DIR=${_llvm}/lib/cmake/llvm"
  "-DClang_DIR=${_llvm}/lib/cmake/clang"
  "-DCMAKE_HIP_ARCHITECTURES=${BPFTIME_LUTHIER_ARCH}"
  "-DCMAKE_CXX_FLAGS=${_rtti_flags} -isystem \"${BPFTIME_ROCM_ROOT}/include\""
  "-DCMAKE_HIP_FLAGS=${_rtti_flags} --rocm-path=\"${BPFTIME_ROCM_ROOT}\" --hip-path=\"${BPFTIME_ROCM_ROOT}\" -isystem \"${BPFTIME_ROCM_ROOT}/include\""
  "-DCMAKE_BUILD_TYPE=Release")

ExternalProject_Add(bpftime_luthier_backend
  SOURCE_DIR "${_source}" BINARY_DIR "${_backend}"
  DEPENDS ${_llvm_source_dependency}
  DOWNLOAD_COMMAND "" UPDATE_COMMAND "" INSTALL_COMMAND ""
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS ${_common_args}
    "-DCMAKE_MAKE_PROGRAM=${BPFTIME_LUTHIER_NINJA}"
    "-DLUTHIER_LLVM_SRC_DIR=${_llvm_source}"
    -DLUTHIER_BUILD_EXAMPLES=OFF -DLUTHIER_BUILD_SHARED_TOOLING=OFF -DLUTHIER_ENABLE_SPIRV=OFF
    -DLUTHIER_LLVM_SPIRV_TRANSLATOR_FOUND=OFF
  BUILD_COMMAND "${CMAKE_COMMAND}" --build "${_backend}" --target
    LuthierToolCXXCompilationPlugin LuthierToolIRCompilationPlugin
    --parallel "${BPFTIME_LUTHIER_JOBS}"
  COMMAND "${CMAKE_COMMAND}" --build "${_backend}" --target LuthierTooling
    --parallel "${BPFTIME_LUTHIER_JOBS}"
  BUILD_ALWAYS TRUE)

ExternalProject_Add(bpftime_luthier_fixtures
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" BINARY_DIR "${_examples}"
  DOWNLOAD_COMMAND "" UPDATE_COMMAND "" INSTALL_COMMAND ""
  DEPENDS bpftime_luthier_backend
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS ${_common_args} "-DCMAKE_MAKE_PROGRAM=${BPFTIME_LUTHIER_NINJA}"
    "-Dluthier_DIR=${_backend}/lib/cmake/luthier"
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
# Building the verifier always builds the independent HIP fixture and tool.
# No GPU work or verification runs until the executable is invoked.
add_dependencies(bpftime_luthier_probe bpftime_luthier_fixtures)
