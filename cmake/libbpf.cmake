#
# Setup libbpf
#
set(LIBBPF_DIR ${CMAKE_CURRENT_LIST_DIR}/../third_party/libbpf/)
include(ExternalProject)
set(LIBBPF_CROSS_ARGS "")
if(CMAKE_CROSSCOMPILING)
  set(LIBBPF_TARGET_CC "${CMAKE_C_COMPILER}")
  if(CMAKE_C_COMPILER_ARG1)
    string(APPEND LIBBPF_TARGET_CC " ${CMAKE_C_COMPILER_ARG1}")
  endif()
  if(CMAKE_SYSROOT)
    string(APPEND LIBBPF_TARGET_CC " --sysroot=${CMAKE_SYSROOT}")
  endif()
  list(APPEND LIBBPF_CROSS_ARGS "CC=${LIBBPF_TARGET_CC}" "AR=${CMAKE_AR}")
endif()
ExternalProject_Add(libbpf
  PREFIX libbpf
  SOURCE_DIR ${LIBBPF_DIR}/src
  CONFIGURE_COMMAND "mkdir" "-p" "${CMAKE_CURRENT_BINARY_DIR}/libbpf/libbpf"
  BUILD_COMMAND "INCLUDEDIR=" "LIBDIR=" "UAPIDIR=" "OBJDIR=${CMAKE_CURRENT_BINARY_DIR}/libbpf/libbpf" "DESTDIR=${CMAKE_CURRENT_BINARY_DIR}/libbpf" "make" ${LIBBPF_CROSS_ARGS} "CFLAGS=-g -O2 -std=gnu89 -fPIC -fvisibility=hidden -DSHARED -DCUSTOM_DEFINE=1" "-j" "install"
  BUILD_IN_SOURCE TRUE
  BUILD_ALWAYS TRUE
  INSTALL_COMMAND ""
  STEP_TARGETS build
  BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/libbpf/libbpf.a
)

# Set BpfObject input parameters -- note this is usually not necessary unless
# you're in a highly vendored environment (like libbpf-bootstrap)
set(LIBBPF_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/libbpf/)
set(LIBBPF_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/libbpf/libbpf.a)


function(copy_header SRC_DIR TARGET_DIR)
  file(GLOB_RECURSE FILES RELATIVE "${SRC_DIR}" "${SRC_DIR}/*")
  message(STATUS "copying ${FILES} from ${SRC_DIR} to ${TARGET_DIR}")

  foreach(file ${FILES})
    get_filename_component(PARENT_DIR "${TARGET_DIR}/${file}" DIRECTORY)
    add_custom_command(
      COMMAND ${CMAKE_COMMAND} -E make_directory ${PARENT_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy
      ${SRC_DIR}/${file}
      ${TARGET_DIR}/${file}
      COMMENT "Copying file ${SRC_DIR}/${file} to ${TARGET_DIR}/${file}"
      OUTPUT ${TARGET_DIR}/${file}
      DEPENDS ${SRC_DIR}/${file}
    )
    list(APPEND header_output_list ${TARGET_DIR}/${file})
    set(header_output_list ${header_output_list} PARENT_SCOPE)
  endforeach()
endfunction()

copy_header("${LIBBPF_DIR}/include/linux" "${LIBBPF_INCLUDE_DIRS}/linux")
copy_header("${LIBBPF_DIR}/include/uapi/" "${LIBBPF_INCLUDE_DIRS}/uapi")
copy_header("${LIBBPF_DIR}/include/uapi/linux" "${LIBBPF_INCLUDE_DIRS}/linux")
add_custom_target(copy_headers ALL
  COMMENT "Copying headers"
  DEPENDS ${header_output_list}
)

message(STATUS "All headers to copy: ${header_output_list}")

set(HEADER_FILES relo_core.h hashmap.h nlattr.h libbpf_internal.h)

foreach(file ${HEADER_FILES})
  add_custom_command(
    TARGET copy_headers
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${LIBBPF_DIR}/src/${file}
    ${LIBBPF_INCLUDE_DIRS}/bpf/${file}
    COMMENT "Copying ${file}"
    BYPRODUCTS ${LIBBPF_INCLUDE_DIRS}/bpf/${file}
  )
endforeach()

add_dependencies(copy_headers libbpf)

add_custom_target(libbpf_with_headers)

add_dependencies(libbpf_with_headers libbpf copy_headers)

# # Setup bpftool
set(BPFTOOL_DIR ${CMAKE_CURRENT_LIST_DIR}/../third_party/bpftool)
set(BPFTOOL_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/bpftool)
ExternalProject_Add(bpftool
  PREFIX bpftool
  SOURCE_DIR ${BPFTOOL_DIR}/src
  CONFIGURE_COMMAND "mkdir" "-p" "${BPFTOOL_INSTALL_DIR}"
  BUILD_COMMAND "make" "EXTRA_CFLAGS=-g -O2 " "-j"
  INSTALL_COMMAND "cp" "${BPFTOOL_DIR}/src/bpftool" "${BPFTOOL_INSTALL_DIR}/bpftool"
  BUILD_IN_SOURCE TRUE
  BUILD_BYPRODUCTS ${BPFTOOL_DIR}/src/bpftool
  INSTALL_BYPRODUCTS ${BPFTOOL_INSTALL_DIR}/bpftool
)

function(add_bpf_skel_generating_target target_name bpf_program output_skel)
  add_custom_command(
    OUTPUT ${output_skel}
    COMMAND "${BPFTOOL_INSTALL_DIR}/bpftool" "gen" "skeleton" "${bpf_program}" > "${output_skel}"
    DEPENDS bpftool ${bpf_program}
  )
  add_custom_target(${target_name}
    DEPENDS ${output_skel}
  )
endfunction()

# Define a helper function
function(add_ebpf_program_target target_name source_file output_file)
  find_program(BPFTIME_BPF_CLANG NAMES clang clang-20 clang-18)
  if(NOT BPFTIME_BPF_CLANG)
    message(FATAL_ERROR "A host clang with the BPF target is required")
  endif()
  # The eBPF program must be built for the architecture it will *run* on, not
  # the one doing the build: __TARGET_ARCH_* and vmlinux.h decide which
  # register offsets PT_REGS_PARM*/PT_REGS_IP read out of pt_regs. Taking the
  # build host here silently produces a program that reads the wrong registers
  # in a cross build. CMAKE_SYSTEM_PROCESSOR is the target architecture when
  # cross compiling and the host architecture otherwise, so native builds keep
  # selecting exactly what `uname -m` used to select.
  set(BPF_TARGET_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
  if(NOT BPF_TARGET_ARCH)
    execute_process(COMMAND uname -m
      OUTPUT_VARIABLE BPF_TARGET_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
  string(TOLOWER "${BPF_TARGET_ARCH}" BPF_TARGET_ARCH)
  if(BPF_TARGET_ARCH MATCHES "^(x86_64|i.86|amd64)$")
    set(UNAME_ARCH_STRIPPED "x86")
  elseif(BPF_TARGET_ARCH MATCHES "^(aarch64|arm64)$")
    set(UNAME_ARCH_STRIPPED "arm64")
  elseif(BPF_TARGET_ARCH MATCHES "^arm")
    set(UNAME_ARCH_STRIPPED "arm")
  elseif(BPF_TARGET_ARCH MATCHES "^(ppc|powerpc)")
    set(UNAME_ARCH_STRIPPED "powerpc")
  elseif(BPF_TARGET_ARCH MATCHES "^mips")
    set(UNAME_ARCH_STRIPPED "mips")
  elseif(BPF_TARGET_ARCH MATCHES "^riscv")
    set(UNAME_ARCH_STRIPPED "riscv")
  elseif(BPF_TARGET_ARCH MATCHES "^loongarch")
    set(UNAME_ARCH_STRIPPED "loongarch")
  else()
    set(UNAME_ARCH_STRIPPED "${BPF_TARGET_ARCH}")
  endif()
  add_custom_command(
    OUTPUT ${output_file}
    COMMAND ${BPFTIME_BPF_CLANG} -O2 -target bpf -c -g -D__TARGET_ARCH_${UNAME_ARCH_STRIPPED} -I${CMAKE_SOURCE_DIR}/third_party/vmlinux/${UNAME_ARCH_STRIPPED} -I${LIBBPF_INCLUDE_DIRS}/uapi -I${LIBBPF_INCLUDE_DIRS} ${source_file} -o ${output_file}
    DEPENDS ${source_file}
  )
  add_custom_target(${target_name}
    DEPENDS ${output_file}
  )
  add_dependencies(${target_name} copy_headers)
endfunction()
