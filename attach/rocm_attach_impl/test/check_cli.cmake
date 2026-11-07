function(check_exit expected)
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env
        --unset=ROCP_TOOL_LIBRARIES --unset=LD_PRELOAD --unset=BPFTIME_ROCM_ATTACH
        --unset=BPFTIME_LOG_OUTPUT --unset=SPDLOG_LEVEL
        "HOME=/proc/bpftime-unwritable-home"
        "${CLI}" start --rocm ${ARGN}
        RESULT_VARIABLE status OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 10)
    if(NOT status STREQUAL "${expected}")
        message(FATAL_ERROR "Unexpected CLI status ${status}, expected ${expected}: ${out}${err}")
    endif()
endfunction()
check_exit(0 /bin/true)
check_exit(7 /bin/sh -c "exit 7")
check_exit(143 /bin/sh -c "kill -TERM $$")
check_exit(127 /bpftime/nonexistent-command)
check_exit(1 --enable-syscall-trace /bin/true)

# /usr/bin/env does not initialize ROCm, so sentinel tool names are sufficient
# to inspect environment composition without loading fake tools into a GPU host.
execute_process(COMMAND "${CMAKE_COMMAND}" -E env
    "ROCP_TOOL_LIBRARIES=/tmp/other-tool.so:${AGENT}"
    "LD_PRELOAD=" "AGENT_SO=keep-me" --unset=BPFTIME_LOG_OUTPUT --unset=SPDLOG_LEVEL
    "HOME=/proc/bpftime-unwritable-home"
    "${CLI}" start --rocm /usr/bin/env
    RESULT_VARIABLE status OUTPUT_VARIABLE out ERROR_VARIABLE err)
foreach(expected "ROCP_TOOL_LIBRARIES=/tmp/other-tool.so:${AGENT}:\n"
    "BPFTIME_ROCM_ATTACH=1\n" "LD_PRELOAD=\n" "AGENT_SO=keep-me\n")
    string(FIND "${out}" "${expected}" position)
    if(NOT status EQUAL 0 OR position LESS 0 OR NOT err STREQUAL "")
        message(FATAL_ERROR "CLI did not preserve the launch environment: ${out}${err}")
    endif()
endforeach()
