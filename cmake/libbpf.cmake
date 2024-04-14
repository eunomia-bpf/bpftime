#
# Setup libbpf
#
set(LIBBPF_DIR ${CMAKE_CURRENT_LIST_DIR}/../third_party/libbpf/)
include(ExternalProject)
ExternalProject_Add(libbpf
  PREFIX libbpf
  SOURCE_DIR ${LIBBPF_DIR}/src
  CONFIGURE_COMMAND "mkdir" "-p" "${CMAKE_CURRENT_BINARY_DIR}/libbpf/libbpf"
  BUILD_COMMAND "INCLUDEDIR=" "LIBDIR=" "UAPIDIR=" "OBJDIR=${CMAKE_CURRENT_BINARY_DIR}/libbpf/libbpf" "DESTDIR=${CMAKE_CURRENT_BINARY_DIR}/libbpf" "make" "CFLAGS=-g -O2 -Werror -Wall -std=gnu89 -fPIC -fvisibility=hidden -DSHARED -DCUSTOM_DEFINE=1" "-j" "install"
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
  BUILD_COMMAND "make" "EXTRA_CFLAGS=-g -O2 -Wall -Werror " "-j"
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
  # opensnoop.bpf.o
  execute_process(COMMAND bash -c "uname -m | sed 's/x86_64/x86/' \
| sed 's/arm.*/arm/' \
| sed 's/aarch64/arm64/' \
| sed 's/ppc64le/powerpc/' \
| sed 's/mips.*/mips/' \
| sed 's/riscv64/riscv/' \
| sed 's/loongarch64/loongarch/'"
    OUTPUT_VARIABLE UNAME_ARCH
  )
  string(STRIP ${UNAME_ARCH} UNAME_ARCH_STRIPPED)
  add_custom_command(
    OUTPUT ${output_file}
    COMMAND clang -Xlinker --export-dynamic -O2 -target bpf -c -g -D__TARGET_ARCH_${UNAME_ARCH_STRIPPED} -I${CMAKE_SOURCE_DIR}/third_party/vmlinux/${UNAME_ARCH_STRIPPED} -I${LIBBPF_INCLUDE_DIRS}/uapi -I${LIBBPF_INCLUDE_DIRS} ${source_file} -o ${output_file}
    DEPENDS ${source_file}
  )
  add_custom_target(${target_name}
    DEPENDS ${output_file}
  )
  add_dependencies(${target_name} copy_headers)
endfunction()
