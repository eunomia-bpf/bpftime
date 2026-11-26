# add frida as external project
set(FRIDA_DOWNLOAD_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/third_party/frida)

set(FRIDA_DOWNLOAD_URL_PREFIX "" CACHE STRING "The prefix added to the frida download url. For example, https://ghproxy.com/")

message(STATUS "System Name: ${CMAKE_SYSTEM_NAME}")
message(STATUS "System Version: ${CMAKE_SYSTEM_VERSION}")
message(STATUS "System Processor: ${CMAKE_SYSTEM_PROCESSOR}")

set(FRIDA_OS_ARCH_RAW "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
string(TOLOWER ${FRIDA_OS_ARCH_RAW} FRIDA_OS_ARCH)
set(FRIDA_VERSION "17.27.6")

message(STATUS "Using frida: arch=${FRIDA_OS_ARCH}, version=${FRIDA_VERSION}")

if(${FRIDA_OS_ARCH} STREQUAL "linux-x86_64")
  set(FRIDA_CORE_DEVKIT_SHA256 "774b1a79aa45513ecad9d293cf1a36f79400d2f67fefb9296840d004a729fd0d")
  set(FRIDA_GUM_DEVKIT_SHA256 "bc03338a2cd34fb33422174a0de1166ab6a108417c0815a7cee20fcf5b332722")
elseif(${FRIDA_OS_ARCH} STREQUAL "linux-aarch64")
  set(FRIDA_CORE_DEVKIT_SHA256 "5255716291fd4349878cf042dd54c2effd494b3e7f608ac42fbc552c897f25af")
  set(FRIDA_GUM_DEVKIT_SHA256 "ea0a2c7532cb7899f377ae54cd73d723b6d918f3357d9a2384e16c15ad1f1329")
  # Cmake uses aarch64, but frida uses arm64
  set(FRIDA_OS_ARCH "linux-arm64")
elseif(${FRIDA_OS_ARCH} MATCHES "linux-arm.*")
  set(FRIDA_CORE_DEVKIT_SHA256 "a524f278f486f75f8242f3c0872b467ddd96c0de554a4ba44f2570e340e929b2")
  set(FRIDA_GUM_DEVKIT_SHA256 "aba872a5eadf7f383819f0507f19260629a4b38144d26776f1f7d3d577aadc7a")
  # Frida only has armhf builds..
  set(FRIDA_OS_ARCH "linux-armhf")
elseif(${FRIDA_OS_ARCH} MATCHES "darwin-arm64")
  set(FRIDA_CORE_DEVKIT_SHA256 "bdc9f647cc0ef985c2cd5953a81e30a318ffcbc9c189ca795526eb48e89be6b4")
  set(FRIDA_GUM_DEVKIT_SHA256 "fe173b81773e34496c9cac6e1bcb117bf272eed8d471f5d665fc6242cee67d9b")
  # for macos-arm m* chip series 
  set(FRIDA_OS_ARCH "macos-arm64")
else()
  message(FATAL_ERROR "Unsupported frida arch ${FRIDA_OS_ARCH}")
endif()

set(FRIDA_CORE_FILE_NAME "frida-core-devkit-${FRIDA_VERSION}-${FRIDA_OS_ARCH}.tar.xz")
set(FRIDA_GUM_FILE_NAME "frida-gum-devkit-${FRIDA_VERSION}-${FRIDA_OS_ARCH}.tar.xz")
set(FRIDA_CORE_DEVKIT_URL "${FRIDA_DOWNLOAD_URL_PREFIX}https://github.com/frida/frida/releases/download/${FRIDA_VERSION}/${FRIDA_CORE_FILE_NAME}")
set(FRIDA_GUM_DEVKIT_URL "${FRIDA_DOWNLOAD_URL_PREFIX}https://github.com/frida/frida/releases/download/${FRIDA_VERSION}/${FRIDA_GUM_FILE_NAME}")

set(FRIDA_CORE_DEVKIT_PATH ${FRIDA_DOWNLOAD_LOCATION}/${FRIDA_CORE_FILE_NAME})
set(FRIDA_GUM_DEVKIT_PATH ${FRIDA_DOWNLOAD_LOCATION}/${FRIDA_GUM_FILE_NAME})

set(FRIDA_CORE_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/FridaCore-prefix/src/FridaCore)
set(FRIDA_GUM_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/FridaGum-prefix/src/FridaGum)

# if file exists, skip download
if(NOT EXISTS ${FRIDA_CORE_DEVKIT_PATH})
  message(STATUS "Downloading Frida Core Devkit")
  set(FRIDA_CORE_DOWNLOAD_URL ${FRIDA_CORE_DEVKIT_URL})
else()
  message(STATUS "Frida Core Devkit already downloaded")
  set(FRIDA_CORE_DOWNLOAD_URL ${FRIDA_CORE_DEVKIT_PATH})
endif()

# if file exists, skip download
if(NOT EXISTS ${FRIDA_GUM_DEVKIT_PATH})
  message(STATUS "Downloading Frida GUM Devkit")
  set(FRIDA_GUM_DOWNLOAD_URL ${FRIDA_GUM_DEVKIT_URL})
else()
  message(STATUS "Frida GUM Devkit already downloaded")
  set(FRIDA_GUM_DOWNLOAD_URL ${FRIDA_GUM_DEVKIT_PATH})
endif()

message(STATUS "Downloading FridaCore from ${FRIDA_CORE_DOWNLOAD_URL}")
include(ExternalProject)
ExternalProject_Add(FridaCore
  URL ${FRIDA_CORE_DOWNLOAD_URL}
  DOWNLOAD_DIR ${FRIDA_DOWNLOAD_LOCATION}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${FRIDA_CORE_INSTALL_DIR}/libfrida-core.a
  URL_HASH SHA256=${FRIDA_CORE_DEVKIT_SHA256}
)

message(STATUS "Downloading FridaGum from ${FRIDA_GUM_DOWNLOAD_URL}")
ExternalProject_Add(FridaGum
  URL ${FRIDA_GUM_DOWNLOAD_URL}
  DOWNLOAD_DIR ${FRIDA_DOWNLOAD_LOCATION}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${FRIDA_GUM_INSTALL_DIR}/libfrida-gum.a
  URL_HASH SHA256=${FRIDA_GUM_DEVKIT_SHA256}
)
