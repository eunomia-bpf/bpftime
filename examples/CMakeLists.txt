project(
  "example"
  LANGUAGES C CXX
)

add_executable(example example.c)

add_subdirectory(../ ../build)
target_link_libraries(example PRIVATE runtime)
target_include_directories(example PRIVATE ../runtime/include)
# 这几行复制到哪里都可以用
