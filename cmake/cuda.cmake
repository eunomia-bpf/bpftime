


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

    set(CUDA_LIBRARY_PATH
        ${BPFTIME_CUDA_ROOT}/targets/${CUDA_TARGET_ARCH}/lib/
        ${BPFTIME_CUDA_ROOT}/targets/${CUDA_TARGET_ARCH}/lib/stubs/
        ${BPFTIME_CUDA_ROOT}/extras/CUPTI/lib64/
        PARENT_SCOPE)

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

    # Prefer canonical CUPTI include path; fall back to versioned layout if needed.
    set(CUDA_CUPTI_INCLUDE_CANDIDATES
        "${BPFTIME_CUDA_ROOT}/extras/CUPTI/include"
        "${BPFTIME_CUDA_ROOT}/cuda-${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}/extras/CUPTI/include")
    set(CUDA_CUPTI_INCLUDE "")
    foreach(_cand IN LISTS CUDA_CUPTI_INCLUDE_CANDIDATES)
        if(EXISTS "${_cand}")
            set(CUDA_CUPTI_INCLUDE "${_cand}")
            break()
        endif()
    endforeach()

    set(CUDA_INCLUDE_PATH
        ${BPFTIME_CUDA_ROOT}/targets/${CUDA_TARGET_ARCH}/include
        ${CUDA_CUPTI_INCLUDE}
        PARENT_SCOPE)

    message(STATUS "Detected CUDA version: ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}")

    # Check if libcupti_static.a exists, prefer static library
    if(EXISTS "${BPFTIME_CUDA_ROOT}/extras/CUPTI/lib64/libcupti_static.a")
        set(CUDA_CUPTI_LIB "libcupti_static.a")
        message(STATUS "CUDA ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}: Using libcupti_static.a")
    else()
        set(CUDA_CUPTI_LIB "cupti")
        message(STATUS "CUDA ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}: Using cupti (dynamic library)")
    endif()

    # nvPTXCompiler API is required by bpftime for PTX compilation.
    # Prefer the static archive if present, otherwise fall back to shared library.
    set(NVPTXCOMPILER_LIB "")
    set(_nvptxcompiler_candidates
        "${BPFTIME_CUDA_ROOT}/targets/${CUDA_TARGET_ARCH}/lib/libnvptxcompiler_static.a"
        "${BPFTIME_CUDA_ROOT}/lib64/libnvptxcompiler_static.a"
        "${BPFTIME_CUDA_ROOT}/targets/${CUDA_TARGET_ARCH}/lib/libnvptxcompiler.so"
        "${BPFTIME_CUDA_ROOT}/lib64/libnvptxcompiler.so")
    foreach(_cand IN LISTS _nvptxcompiler_candidates)
        if(EXISTS "${_cand}")
            set(NVPTXCOMPILER_LIB "${_cand}")
            break()
        endif()
    endforeach()

    if(NVPTXCOMPILER_LIB)
        set(CUDA_LIBS cuda cudart ${NVPTXCOMPILER_LIB} ${CUDA_CUPTI_LIB} nvrtc)
        message(STATUS "CUDA ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}: Using nvptxcompiler: ${NVPTXCOMPILER_LIB}")
    else()
        set(CUDA_LIBS cuda cudart ${CUDA_CUPTI_LIB} nvrtc)
        message(WARNING "CUDA ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}: nvptxcompiler library not found; targets using nvPTXCompiler API may fail to link")
    endif()

    set(CUDA_LIBS ${CUDA_LIBS} PARENT_SCOPE)
endfunction()
