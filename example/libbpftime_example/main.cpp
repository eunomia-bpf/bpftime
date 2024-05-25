#include "bpftime_shm.hpp"
#include <iostream>

int main() {
    bpftime_initialize_global_shm(bpftime::shm_open_type::SHM_CREATE_OR_OPEN);
    const char* jsonFileName = "example_shm.json";
    SPDLOG_INFO("GLOBAL memory initialized ");
    bpftime_export_global_shm_to_json(jsonFileName);
    return 0;
}
