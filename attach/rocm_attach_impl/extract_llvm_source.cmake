# Only headers and TableGen inputs are needed; LLVM is never built here.
cmake_minimum_required(VERSION 3.23)
set(_root "llvm-project-${REVISION}")
file(ARCHIVE_EXTRACT INPUT "${ARCHIVE}" DESTINATION "${DESTINATION}"
  PATTERNS "${_root}/llvm/include/*"
           "${_root}/llvm/lib/Target/AMDGPU/*"
           "${_root}/llvm/utils/TableGen/*")
foreach(_file IN ITEMS llvm/lib/Target/AMDGPU/AMDGPU.td
    llvm/include/llvm/IR/Intrinsics.td llvm/utils/TableGen/Common/CodeGenTarget.h)
  if(NOT EXISTS "${DESTINATION}/${_root}/${_file}")
    message(FATAL_ERROR "Downloaded LLVM source archive is missing ${_file}")
  endif()
endforeach()
