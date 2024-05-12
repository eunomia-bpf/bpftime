# Utilities to create archive file from static libraries
# Courtesy: https://github.com/WasmEdge/WasmEdge/blob/f5cf26c66e4bbb9bca00497fce5c814aac7d56fa/lib/api/CMakeLists.txt#L39
#
#
# Helper function to construct commands and dependencies.
get_filename_component(CMAKE_AR_NAME "${CMAKE_AR}" NAME)
function(bpftime_add_static_lib_component_command target)
  if(APPLE)
    if(CMAKE_AR_NAME STREQUAL "ar")
      list(APPEND CMDS
        COMMAND ${CMAKE_COMMAND} -E make_directory objs/${target}
        COMMAND ${CMAKE_AR} -x $<TARGET_FILE:${target}>
        COMMAND ${CMAKE_AR} -t $<TARGET_FILE:${target}> | xargs -I '{}' mv '{}' objs/${target}
      )
      set(BPFTIME_STATIC_LIB_AR_CMDS ${BPFTIME_STATIC_LIB_AR_CMDS} ${CMDS} PARENT_SCOPE)
    elseif(CMAKE_AR_NAME STREQUAL "libtool")
      set(BPFTIME_STATIC_LIB_LIBTOOL_FILES ${BPFTIME_STATIC_LIB_LIBTOOL_FILES} $<TARGET_FILE:${target}> PARENT_SCOPE)
    endif()
  else()
    list(APPEND CMDS
      COMMAND ${CMAKE_COMMAND} -E make_directory objs/${target}
      COMMAND ${CMAKE_COMMAND} -E chdir objs/${target} ${CMAKE_AR} -x $<TARGET_FILE:${target}>
    )
    set(BPFTIME_STATIC_LIB_AR_CMDS ${BPFTIME_STATIC_LIB_AR_CMDS} ${CMDS} PARENT_SCOPE)
  endif()
  set(BPFTIME_STATIC_LIB_DEPS ${BPFTIME_STATIC_LIB_DEPS} ${target} PARENT_SCOPE)
endfunction()


# Helper function to construct commands about packaging llvm and dependency libraries with paths.
function(bpftime_add_libs_component_command target_path)
  get_filename_component(target_name ${target_path} NAME)
  string(REGEX REPLACE "^lib" "" target_name ${target_name})
  string(REGEX REPLACE "\.a$" "" target_name ${target_name})
  if(APPLE)
    get_filename_component(CMAKE_AR_NAME "${CMAKE_AR}" NAME)
    if(CMAKE_AR_NAME STREQUAL "ar")
      list(APPEND CMDS
        COMMAND ${CMAKE_COMMAND} -E make_directory objs/${target_name}
        COMMAND ${CMAKE_AR} -x ${target_path}
        COMMAND ${CMAKE_AR} -t ${target_path} | xargs -I '{}' mv '{}' objs/${target_name}
      )
      set(BPFTIME_STATIC_LLVM_LIB_AR_CMDS ${BPFTIME_STATIC_LLVM_LIB_AR_CMDS} ${CMDS} PARENT_SCOPE)
    elseif(CMAKE_AR_NAME STREQUAL "libtool")
      set(BPFTIME_STATIC_LIB_LIBTOOL_FILES ${BPFTIME_STATIC_LIB_LIBTOOL_FILES} ${target_path} PARENT_SCOPE)
    endif()
  else()
    list(APPEND CMDS
      COMMAND ${CMAKE_COMMAND} -E make_directory objs/${target_name}
      COMMAND ${CMAKE_COMMAND} -E chdir objs/${target_name} ${CMAKE_AR} -x ${target_path}
    )
    set(BPFTIME_STATIC_LLVM_LIB_AR_CMDS ${BPFTIME_STATIC_LLVM_LIB_AR_CMDS} ${CMDS} PARENT_SCOPE)
  endif()
endfunction()
