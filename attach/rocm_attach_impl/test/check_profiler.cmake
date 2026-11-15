execute_process(COMMAND "${CMAKE_COMMAND}" -E env
    "LD_LIBRARY_PATH=${MOCK_DIR}:$ENV{LD_LIBRARY_PATH}"
    --unset=ROCP_TOOL_LIBRARIES --unset=LD_PRELOAD --unset=BPFTIME_ROCM_ATTACH
    --unset=BPFTIME_LOG_OUTPUT --unset=SPDLOG_LEVEL --unset=BPFTIME_USED
    "HOME=/proc/bpftime-unwritable-home"
    "${TEST}" "${AGENT}" "${MOCK_DIR}/librocprofiler-sdk.so"
    RESULT_VARIABLE status OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 25)
if(NOT status EQUAL 0 OR NOT out STREQUAL "" OR NOT err STREQUAL "")
    message(FATAL_ERROR "Profiler regression failed (${status}): stdout=${out}, stderr=${err}")
endif()
