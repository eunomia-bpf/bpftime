add_library(ptxpass_core STATIC src/core.cpp)
target_include_directories(ptxpass_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/third_party)
set_property(TARGET ptxpass_core PROPERTY CXX_STANDARD 20)




