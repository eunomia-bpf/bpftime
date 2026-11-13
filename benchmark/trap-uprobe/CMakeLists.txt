if(BPFTIME_ENABLE_TRAP_UPROBE)
    add_executable(trap-uprobe-bench trap_bench.cpp)
    add_dependencies(trap-uprobe-bench bpftime_trap_uprobe_attach_impl)
    target_link_libraries(trap-uprobe-bench
        PRIVATE bpftime_trap_uprobe_attach_impl spdlog::spdlog pthread)
    target_include_directories(trap-uprobe-bench
        PRIVATE ${TRAP_UPROBE_ATTACH_IMPL_INCLUDE}
                ${CMAKE_SOURCE_DIR}/attach/trap_uprobe_attach_impl/src)
    set_property(TARGET trap-uprobe-bench PROPERTY CXX_STANDARD 20)
endif()
