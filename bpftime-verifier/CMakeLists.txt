add_library(bpftime-verifier STATIC src/bpftime-verifier.cpp)
add_subdirectory(ebpf-verifier)

target_include_directories(bpftime-verifier
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ebpf-verifier/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(BPFTIME_VERIFIER_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
