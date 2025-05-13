add_executable(mix_schedule_cpu ./cpu_runner.cpp)

include(../../../cmake/cuda.cmake)
find_cuda()


find_package (Python COMPONENTS Development)

add_dependencies(mix_schedule_cpu runtime Python::Python spdlog::spdlog)


target_link_directories(mix_schedule_cpu PRIVATE ${CUDA_LIBRARY_PATH})
target_link_libraries(mix_schedule_cpu runtime ${CUDA_LIBS} Python::Python  spdlog::spdlog)
target_include_directories(mix_schedule_cpu PRIVATE ${CUDA_INCLUDE_PATH} ${Python_INCLUDE_DIRS} ${SPDLOG_INCLUDE})

target_compile_definitions(mix_schedule_cpu PRIVATE SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR})
