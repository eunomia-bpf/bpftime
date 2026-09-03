# Cross compilation toolchain for riscv64 Linux (rv64gc, lp64d)
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_VERSION 1)
SET(CMAKE_SYSTEM_PROCESSOR riscv64)

# specify the cross compiler
SET(CMAKE_C_COMPILER riscv64-linux-gnu-gcc)
SET(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)
SET(ARCH "riscv64")

# where is the target environment
SET(CMAKE_FIND_ROOT_PATH /usr/riscv64-linux-gnu)

# search for programs in the build host directories
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
