# add frida as external project
set(FRIDA_DOWNLOAD_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/third_party/frida)

set(FRIDA_DOWNLOAD_URL_PREFIX "" CACHE STRING "The prefix added to the frida download url. For example, https://ghproxy.com/")

message(STATUS "System Name: ${CMAKE_SYSTEM_NAME}")
message(STATUS "System Version: ${CMAKE_SYSTEM_VERSION}")
message(STATUS "System Processor: ${CMAKE_SYSTEM_PROCESSOR}")

set(FRIDA_OS_ARCH_RAW "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
string(TOLOWER ${FRIDA_OS_ARCH_RAW} FRIDA_OS_ARCH)
set(FRIDA_VERSION "16.1.2")

message(STATUS "Using frida: arch=${FRIDA_OS_ARCH}, version=${FRIDA_VERSION}")

if(${FRIDA_OS_ARCH} STREQUAL "linux-x86_64")
  set(FRIDA_CORE_DEVKIT_SHA256 "45a7e47c4181662611ce89c2891e5327676aaaf9e53ee205ef98f8d58a312a00")
  set(FRIDA_GUM_DEVKIT_SHA256 "c20af106e089bbbdb9ed4d5dfc63ce9ae8f6643bbb76a6b0afb196726d9a241a")
elseif(${FRIDA_OS_ARCH} STREQUAL "linux-aarch64")
  set(FRIDA_CORE_DEVKIT_SHA256 "38cfb73cf29c6a09918a86b342dc5eab59d311640ea2da5c5904f16cde9e1430")
  set(FRIDA_GUM_DEVKIT_SHA256 "b8cdf63bfb9771320439b21fc97482c90bc230f95ae4b26e42a02393e6e85804")
  # Cmake uses aarch64, but frida uses arm64
  set(FRIDA_OS_ARCH "linux-arm64")
elseif(${FRIDA_OS_ARCH} MATCHES "linux-arm.*")
  set(FRIDA_CORE_DEVKIT_SHA256 "b9b4af5f75d7261ed493fd91b97f2b17345c1cde31622e1250e7d78cd0ff2356")
  set(FRIDA_GUM_DEVKIT_SHA256 "6b5963eb740062aec6c22c46ec2944a68006f72d290f714fb747ffe75b448a60")
  # Frida only has armhf builds..
  set(FRIDA_OS_ARCH "linux-armhf")
elseif(${FRIDA_OS_ARCH} MATCHES "darwin-arm64")
  set(FRIDA_CORE_DEVKIT_SHA256 "7811e516e6b7bbc0153d30095560e0b1133f154060c5542764100d3e0eb2ab2b")
  set(FRIDA_GUM_DEVKIT_SHA256 "03f6085ae5330cf38e0a498784500675fc5bd7361bb551a9097ba5fe397aceda")
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
