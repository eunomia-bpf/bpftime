add_executable(pos_unit_tests
  ./test_fatbin_parse.cpp

)

add_dependencies(pos_unit_tests pos  Catch2)

target_link_libraries(pos_unit_tests PRIVATE pos Catch2::Catch2WithMain )
target_include_directories(pos_unit_tests PRIVATE ${Catch2_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_compile_definitions(pos_unit_tests PRIVATE FATBIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/test.bin)
