#include <iostream>
#include "bpftime_shm.hpp"
#include "spdlog/spdlog.h"

int main() {
    bpftime_initialize_global_shm(bpftime::shm_open_type::SHM_CREATE_OR_OPEN);
    const char* jsonFileName = "example_shm.json";
    const char* importJsonName = "ebpf.json";
    SPDLOG_INFO("GLOBAL memory initialized ");
    // load json program to shm
    bpftime_import_global_shm_from_json(importJsonName);
    // export it to another json file
    bpftime_export_global_shm_to_json(jsonFileName);
    // remove content from global shm
    bpftime_remove_global_shm();
    return 0;
}