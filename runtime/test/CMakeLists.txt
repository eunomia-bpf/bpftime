cmake_minimum_required(VERSION 3.15)

#
# Project details
#
project(
  ${CMAKE_PROJECT_NAME}Tests
  LANGUAGES C
)

set(test_sources

  # probe tests
  # src/test_attach_replace.cpp

  # ufunc tests
  # src/test_ufunc_register.cpp

  # src/test_shm_hash_maps.cpp
  # src/test_shm_progs_attach.cpp

  # src/test_shm_client.cpp
  # src/test_shm_server.cpp
  # src/test_ufunc.cpp
  # src/test_helpers.cpp
)

message(DEBUG "Adding tests under ${CMAKE_PROJECT_NAME}Tests...")

set(test_include_dirs
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../vm/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party
  ${CMAKE_CURRENT_SOURCE_DIR}/../object
)

foreach(file ${test_sources})
  string(REGEX REPLACE "(.*/)([a-zA-Z0-9_ ]+)(\.cpp)" "\\2" test_name ${file})
  add_executable(${test_name}_Tests ${file})
  add_dependencies(${test_name}_Tests runtime bpftime-object)
  target_link_libraries(
    ${test_name}_Tests PRIVATE
    -lm
  )

  if(NOT BPFTIME_ENABLE_ASAN)
    # set the -static flag for static linking
    # set the -static flag for static linking
    # set_target_properties(${test_name}_Tests PROPERTIES LINK_FLAGS "-static")
    # need on qemu-user
  endif()

  #
  # Set the compiler standard
  #

  # target_compile_features(${test_name}_Tests PUBLIC cxx_std_17)
  target_include_directories(${test_name}_Tests
    PRIVATE
    ${test_include_dirs}
  )

  #
  # Setup code coverage if enabled
  #
  if(BPFTIME_ENABLE_CODE_COVERAGE)
    target_compile_options(runtime PUBLIC -O0 -g -fprofile-arcs -ftest-coverage)
    target_link_options(runtime PUBLIC -fprofile-arcs -ftest-coverage)
    message(DEBUG "Code coverage is enabled and provided with GCC.")
  endif()

  #
  # Load the desired unit testing framework
  #
  # Currently supported: GoogleTest (and GoogleMock), Catch2.
  if(${CMAKE_PROJECT_NAME}_USE_CATCH2)
    find_package(Catch2 REQUIRED)
    target_link_libraries(
      ${test_name}_Tests
      PUBLIC
      Catch2::Catch2
      runtime
      bpftime-object
    )
  else()
    target_link_libraries(
      ${test_name}_Tests
      PUBLIC
      runtime
      bpftime-object
    )

    # message("Unknown testing library ${test_name}_Tests. Please setup your desired unit testing library by using `target_link_libraries`.")
  endif()

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Enable sanitizers if we are debugging
    target_link_options(${test_name}_Tests PRIVATE -fsanitize=undefined)
  endif()

  #
  # Add the unit tests
  #
  add_test(
    NAME
    ${test_name}
    COMMAND
    ${test_name}_Tests
  )
endforeach()

message(DEBUG "Finished adding unit tests for ${CMAKE_PROJECT_NAME}.")
