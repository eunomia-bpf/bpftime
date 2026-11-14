# Frida Gum / Core for QNX — official GitHub releases do NOT ship qnx-arm64
# prebuilt devkits. Point BPFTIME_FRIDA_QNX_ROOT at a locally built tree:
#
#   ${BPFTIME_FRIDA_QNX_ROOT}/
#     gum/libfrida-gum.a
#     gum/frida-gum.h          (and related headers)
#     core/libfrida-core.a
#     core/frida-core.h
#
# Build Frida for QNX (on Linux host with QNX SDP), e.g.:
#   make -f Makefile.linux.mk FRIDA_HOST=qnx-arm64
# See: https://github.com/frida/frida/issues/25

set(BPFTIME_FRIDA_QNX_ROOT "" CACHE PATH
  "Root of self-built Frida for QNX aarch64 (gum/ + core/ subdirs)")

if(NOT BPFTIME_FRIDA_QNX_ROOT)
  message(FATAL_ERROR
    "BPFTIME_FRIDA_QNX_ROOT is required when BPFTIME_TARGET_QNX=ON.\n"
    "Build Frida with FRIDA_HOST=qnx-arm64 and set this path to the output tree.")
endif()

set(FRIDA_GUM_INSTALL_DIR "${BPFTIME_FRIDA_QNX_ROOT}/gum")
set(FRIDA_CORE_INSTALL_DIR "${BPFTIME_FRIDA_QNX_ROOT}/core")

if(NOT EXISTS "${FRIDA_GUM_INSTALL_DIR}/libfrida-gum.a")
  message(FATAL_ERROR "Missing ${FRIDA_GUM_INSTALL_DIR}/libfrida-gum.a")
endif()
if(NOT EXISTS "${FRIDA_CORE_INSTALL_DIR}/libfrida-core.a")
  message(FATAL_ERROR "Missing ${FRIDA_CORE_INSTALL_DIR}/libfrida-core.a")
endif()

# Keep ExternalProject target names so existing add_dependencies(... FridaGum)
# and add_dependencies(... FridaCore) continue to work.
add_custom_target(FridaGum)
add_custom_target(FridaCore)

message(STATUS "Frida QNX gum:  ${FRIDA_GUM_INSTALL_DIR}")
message(STATUS "Frida QNX core: ${FRIDA_CORE_INSTALL_DIR}")
