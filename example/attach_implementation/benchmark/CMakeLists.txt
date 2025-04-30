add_subdirectory(ebpf_controller)
add_subdirectory(baseline_nginx_plugin)

# Function to find all subdirectories with Makefiles
function(find_makefile_subdirs result base_dir)
    file(GLOB_RECURSE makefiles "${base_dir}/*/Makefile")
    set(dirs "")
    foreach(makefile ${makefiles})
        get_filename_component(dir ${makefile} DIRECTORY)
        list(APPEND dirs ${dir})
    endforeach()
    set(${result} ${dirs} PARENT_SCOPE)
endfunction()

# Find all subdirectories with Makefiles
find_makefile_subdirs(MAKEFILE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})

# Add targets for each Makefile directory
foreach(dir ${MAKEFILE_DIRS})
    get_filename_component(target_name ${dir} NAME)
    add_custom_target(${target_name}
        COMMAND $(MAKE) -C ${dir}
        COMMENT "Building ${target_name} with its own Makefile"
        VERBATIM
    )
endforeach()
