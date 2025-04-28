cmake_minimum_required(VERSION 3.10)
project(nginx_baseline_controller)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the executable
add_executable(nginx_baseline_controller baseline_controller.cpp)

# Link against required libraries
target_link_libraries(nginx_baseline_controller
    rt    # For shared memory functions (shm_open, etc.)
    pthread
)

# Install the executable
install(TARGETS nginx_baseline_controller
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
) 