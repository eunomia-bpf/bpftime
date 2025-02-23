
find_package(absl REQUIRED)
find_package(protobuf CONFIG REQUIRED)

add_library(apicxt-objects OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/include/proto/apicxt.proto")

target_link_libraries(apicxt-objects PUBLIC protobuf::libprotobuf)

set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/include/proto")

target_include_directories(apicxt-objects PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}>")

protobuf_generate(
        TARGET apicxt-objects
        IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include/proto"
        PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")


add_library(handle-objects OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/include/proto/handle.proto")

target_link_libraries(handle-objects PUBLIC protobuf::libprotobuf)

set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/include/proto")

target_include_directories(handle-objects PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}>")

protobuf_generate(
        TARGET handle-objects
        IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include/proto"
        PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")



add_library(client-objects OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/include/proto/client.proto")

target_link_libraries(client-objects PUBLIC protobuf::libprotobuf)

set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/include/proto")

target_include_directories(client-objects PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}>")

protobuf_generate(
        TARGET client-objects
        IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include/proto"
        PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")

add_library(context-objects OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/cuda_impl/proto/context.proto")

target_link_libraries(context-objects PUBLIC protobuf::libprotobuf)

set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/cuda_impl/proto")

target_include_directories(context-objects PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}> ${CMAKE_CURRENT_BINARY_DIR}/../")

protobuf_generate(
        TARGET context-objects
        IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_impl/proto" IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../"
        PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")


add_library(cublas-objects OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/cuda_impl/proto/cublas.proto")

target_link_libraries(cublas-objects PUBLIC protobuf::libprotobuf)

set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/cuda_impl/proto")

target_include_directories(cublas-objects PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}> ${CMAKE_CURRENT_BINARY_DIR}/../")

protobuf_generate(
        TARGET cublas-objects
        IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_impl/proto" IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../"
        PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")


add_library(device-objects OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/cuda_impl/proto/device.proto")

target_link_libraries(device-objects PUBLIC protobuf::libprotobuf)

set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/cuda_impl/proto")

target_include_directories(device-objects PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}> ${CMAKE_CURRENT_BINARY_DIR}/../")

protobuf_generate(
        TARGET device-objects
        IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_impl/proto" IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../"
        PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")


add_library(event-objects OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/cuda_impl/proto/event.proto")

target_link_libraries(event-objects PUBLIC protobuf::libprotobuf)

set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/cuda_impl/proto")

target_include_directories(event-objects PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}> ${CMAKE_CURRENT_BINARY_DIR}/../")

protobuf_generate(
        TARGET event-objects
        IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_impl/proto" IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../"
        PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")


add_library(function-objects OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/cuda_impl/proto/function.proto")

target_link_libraries(function-objects PUBLIC protobuf::libprotobuf)

set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/cuda_impl/proto")

target_include_directories(function-objects PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}> ${CMAKE_CURRENT_BINARY_DIR}/../")

protobuf_generate(
        TARGET function-objects
        IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_impl/proto" IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../"
        PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")


add_library(memory-objects OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/cuda_impl/proto/memory.proto")

target_link_libraries(memory-objects PUBLIC protobuf::libprotobuf)

set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/cuda_impl/proto")

target_include_directories(memory-objects PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}> ${CMAKE_CURRENT_BINARY_DIR}/../")

protobuf_generate(
        TARGET memory-objects
        IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_impl/proto" IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../"
        PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")


add_library(module-objects OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/cuda_impl/proto/module.proto")

target_link_libraries(module-objects PUBLIC protobuf::libprotobuf)

set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/cuda_impl/proto")

target_include_directories(module-objects PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}> ${CMAKE_CURRENT_BINARY_DIR}/../")

protobuf_generate(
        TARGET module-objects
        IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_impl/proto" IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../"
        PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")


add_library(stream-objects OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/cuda_impl/proto/stream.proto")

target_link_libraries(stream-objects PUBLIC protobuf::libprotobuf)

set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/cuda_impl/proto")

target_include_directories(stream-objects PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}> ${CMAKE_CURRENT_BINARY_DIR}/../")

protobuf_generate(
        TARGET stream-objects
        IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_impl/proto" IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../"
        PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")


add_library(var-objects OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/cuda_impl/proto/var.proto")

target_link_libraries(var-objects PUBLIC protobuf::libprotobuf)

set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/cuda_impl/proto")

target_include_directories(var-objects PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}> ${CMAKE_CURRENT_BINARY_DIR}/../")

protobuf_generate(
        TARGET var-objects
        IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_impl/proto" IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../"
        PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")
message(${CMAKE_CURRENT_SOURCE_DIR}/../)

file(GLOB_RECURSE SOURCE_FILES src/*.cpp src/oob/*.cpp cuda_impl/*.cpp cuda_impl/*/*.cpp)
add_library(pos ${SOURCE_FILES})
target_include_directories(pos PRIVATE ${CMAKE_SOURCE_DIR}/runtime ${CMAKE_CURRENT_BINARY_DIR}/../)
target_link_libraries(pos PRIVATE absl::strings absl::synchronization absl::flat_hash_map absl::time absl::base protobuf::libprotobuf cublas-objects device-objects event-objects function-objects memory-objects module-objects stream-objects var-objects context-objects cublas-objects apicxt-objects handle-objects client-objects cuda cublas elf ibverbs /usr/lib/llvm-14/lib/libclang.so)
set(CMAKE_CXX_FLAGS "-I/home/geekpie/miniconda3/envs/asc25-rna/include/ -I/usr/lib/llvm-14/include/ -I${CMAKE_CURRENT_BINARY_DIR}/../")