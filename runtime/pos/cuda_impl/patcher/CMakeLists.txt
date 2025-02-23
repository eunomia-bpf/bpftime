# ref: https://github.com/XiangpengHao/cxx-cmake-example

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_CMD cargo build)
    set(TARGET_DIR "debug")
else ()
    set(CARGO_CMD cargo build --release)
    set(TARGET_DIR "release")
endif ()

set(RUST_CXX_LIB ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_DIR}/libpatcher.a)
set(RUST_CXX_SRC ${CMAKE_CURRENT_BINARY_DIR}/patcher.cc)
set(RUST_CXX_INC ${CMAKE_CURRENT_BINARY_DIR})

file(GLOB RUST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.rs)

add_custom_command(
    OUTPUT patcher.cc
    COMMAND CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR} ${CARGO_CMD}
    # copy generated src for make clean to delete
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/cxxbridge/patcher/src/lib.rs.cc ${RUST_CXX_SRC}
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/cxxbridge/patcher/src/lib.rs.h ${CMAKE_CURRENT_BINARY_DIR}/patcher.h
    DEPENDS ${RUST_SRC}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(patcher STATIC ${RUST_CXX_SRC})
target_link_libraries(patcher PRIVATE ${RUST_CXX_LIB})
target_include_directories(patcher INTERFACE ${RUST_CXX_INC})
