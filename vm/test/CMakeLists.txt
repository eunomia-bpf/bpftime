cmake_minimum_required(VERSION 3.15)

#
# Project details
#
project(
  ${CMAKE_PROJECT_NAME}Tests
  LANGUAGES C
)

add_subdirectory(bpf_conformance_runner)

message(DEBUG "Adding tests under ${CMAKE_PROJECT_NAME}Tests...")

foreach(file ${test_sources})
  string(REGEX REPLACE "(.*/)([a-zA-Z0-9_ ]+)(\.c)" "\\2" test_name ${file})
  add_executable(${test_name}_Tests ${file})
  target_link_libraries(
    ${test_name}_Tests PRIVATE
    -lm)

  if(NOT BPFTIME_ENABLE_ASAN)
    # set the -static flag for static linking
    # set_target_properties(${test_name}_Tests PROPERTIES LINK_FLAGS "-static")
    # need on qemu-user
  endif()

  add_dependencies(${test_name}_Tests bpftime_vm)
  #
  # Set the compiler standard
  #

  # target_compile_features(${test_name}_Tests PUBLIC cxx_std_17)
  target_include_directories(${test_name}_Tests
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  #
  # Setup code coverage if enabled
  #
  if(BPFTIME_ENABLE_CODE_COVERAGE)
    target_compile_options(bpftime_vm_compat PUBLIC -O0 -g -fprofile-arcs -ftest-coverage)
    target_link_options(bpftime_vm_compat PUBLIC -fprofile-arcs -ftest-coverage)
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
      bpftime_vm
    )
  else()
    target_link_libraries(
      ${test_name}_Tests
      PUBLIC
      bpftime_vm
    )
    # message("Unknown testing library ${test_name}_Tests. Please setup your desired unit testing library by using `target_link_libraries`.")
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
