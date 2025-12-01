


function(find_cuda)
    if(NOT BPFTIME_CUDA_ROOT)
        message(FATAL_ERROR "To use NV attach, set BPFTIME_CUDA_ROOT to the root of CUDA installation, such as /usr/local/cuda-12.6")
    endif()

    # Detect target platform based on CMAKE_SYSTEM_PROCESSOR
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(CUDA_TARGET_ARCH "aarch64-linux")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
        set(CUDA_TARGET_ARCH "x86_64-linux")
    else()
        message(WARNING "Unsupported architecture ${CMAKE_SYSTEM_PROCESSOR}, defaulting to x86_64-linux")
        set(CUDA_TARGET_ARCH "x86_64-linux")
    endif()

    set(CUDA_LIBRARY_PATH ${BPFTIME_CUDA_ROOT}/targets/${CUDA_TARGET_ARCH}/lib/ ${BPFTIME_CUDA_ROOT}/targets/${CUDA_TARGET_ARCH}/lib/stubs/ ${BPFTIME_CUDA_ROOT}/extras/CUPTI/lib64/ PARENT_SCOPE)

    # Detect CUDA version from version.json or version.txt
    if(EXISTS "${BPFTIME_CUDA_ROOT}/version.json")
        file(READ "${BPFTIME_CUDA_ROOT}/version.json" CUDA_VERSION_JSON)
        # Match the cuda section and extract version number (e.g., "12.8.1" -> major=12, minor=8)
        string(REGEX MATCH "\"cuda\"[^{]*\\{[^}]*\"version\"[^\"]*\"([0-9]+)\\.([0-9]+)" _ "${CUDA_VERSION_JSON}")
        set(CUDA_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(CUDA_VERSION_MINOR ${CMAKE_MATCH_2})
    elseif(EXISTS "${BPFTIME_CUDA_ROOT}/version.txt")
        file(READ "${BPFTIME_CUDA_ROOT}/version.txt" CUDA_VERSION_TXT)
        string(REGEX MATCH "CUDA Version ([0-9]+)\\.([0-9]+)" _ "${CUDA_VERSION_TXT}")
        set(CUDA_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(CUDA_VERSION_MINOR ${CMAKE_MATCH_2})
    else()
        # Try to extract from path name as fallback (e.g., cuda-13.0)
        string(REGEX MATCH "cuda-([0-9]+)\\.([0-9]+)" _ "${BPFTIME_CUDA_ROOT}")
        set(CUDA_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(CUDA_VERSION_MINOR ${CMAKE_MATCH_2})
    endif()

    # CUPTI include path should be dynamic based on detected CUDA version
    set(CUDA_INCLUDE_PATH ${BPFTIME_CUDA_ROOT}/targets/${CUDA_TARGET_ARCH}/include ${BPFTIME_CUDA_ROOT}/cuda-${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}/extras/CUPTI/include PARENT_SCOPE)

    message(STATUS "Detected CUDA version: ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}")

    # Check if libcupti_static.a exists, prefer static library
    if(EXISTS "${BPFTIME_CUDA_ROOT}/extras/CUPTI/lib64/libcupti_static.a")
        set(CUDA_CUPTI_LIB "libcupti_static.a")
        message(STATUS "CUDA ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}: Using libcupti_static.a")
    else()
        set(CUDA_CUPTI_LIB "cupti")
        message(STATUS "CUDA ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}: Using cupti (dynamic library)")
    endif()

    # nvptxcompiler_static is only available in CUDA 12.x and earlier
    if(CUDA_VERSION_MAJOR LESS 13)
        set(CUDA_LIBS cuda cudart libnvptxcompiler_static.a ${CUDA_CUPTI_LIB} nvrtc PARENT_SCOPE)
        message(STATUS "CUDA ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}: Including nvptxcompiler_static")
    else()
        set(CUDA_LIBS cuda cudart ${CUDA_CUPTI_LIB} nvrtc PARENT_SCOPE)
        message(STATUS "CUDA ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}: Excluding nvptxcompiler_static (not available)")
    endif()
endfunction()
