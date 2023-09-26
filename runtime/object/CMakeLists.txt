
add_library(bpftime-object OBJECT
  bpf_object.cpp
)

#
# Setup libbpf
#
set(LIBBPF_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/libbpf/)
include(ExternalProject)
ExternalProject_Add(libbpf
  PREFIX libbpf
  SOURCE_DIR ${LIBBPF_DIR}/src
  CONFIGURE_COMMAND "mkdir" "-p" "${CMAKE_CURRENT_BINARY_DIR}/libbpf/libbpf"
  BUILD_COMMAND "INCLUDEDIR=" "LIBDIR=" "UAPIDIR=" "OBJDIR=${CMAKE_CURRENT_BINARY_DIR}/libbpf/libbpf" "DESTDIR=${CMAKE_CURRENT_BINARY_DIR}/libbpf" "make" "CFLAGS=-g -O2 -Werror -Wall -std=gnu89 -fPIC -fvisibility=hidden -DSHARED -DCUSTOM_DEFINE=1" "-j" "install"
  BUILD_IN_SOURCE TRUE
  INSTALL_COMMAND ""
  STEP_TARGETS build
  BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/libbpf/libbpf.a
)

# Set BpfObject input parameters -- note this is usually not necessary unless
# you're in a highly vendored environment (like libbpf-bootstrap)
set(LIBBPF_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/libbpf)
set(LIBBPF_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/libbpf/libbpf.a)

#
add_custom_target(copy_headers ALL
  COMMENT "Copying headers"
)

function(copy_header TARGET SRC_DIR TARGET_DIR)
  file(GLOB_RECURSE FILES RELATIVE "${SRC_DIR}" "${SRC_DIR}/*")
  message(STATUS "copying ${FILES} from ${SRC_DIR} to ${TARGET_DIR}")

  foreach(file ${FILES})
    get_filename_component(PARENT_DIR "${TARGET_DIR}/${file}" DIRECTORY)
    add_custom_command(
      TARGET ${TARGET}
      PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory ${PARENT_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy
      ${SRC_DIR}/${file}
      ${TARGET_DIR}/${file}
      COMMENT "Copying file ${HEADER_DIRS}/${file} to ${TARGET_DIR}/${file}"
      BYPRODUCTS ${TARGET_DIR}/${file}
    )
  endforeach()
endfunction()

copy_header(copy_headers "${LIBBPF_DIR}/include/linux" "${LIBBPF_INCLUDE_DIRS}/linux")
copy_header(copy_headers "${LIBBPF_DIR}/include/uapi/linux" "${LIBBPF_INCLUDE_DIRS}/linux")

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
