#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <memory>
#include <bpftime_shm.hpp>
#include <ostream>
#include <string>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <spdlog/cfg/env.h>
#include <variant>
#include <vector>
#include "bpftime_helper_group.hpp"
#include "bpftime_prog.hpp"
#include "bpftime_shm.hpp"
#include "bpf_attach_ctx.hpp"
#include <bpftime_shm_internal.hpp>
#include <fstream>
#include <time.h>
#include <iostream>
#include <chrono>

using namespace std;
using namespace bpftime;

static int run_ebpf_and_measure(bpftime_prog& prog, std::vector<uint8_t>& data_in, int repeat_N) {
    // Test the program
    uint64_t return_val;
    void* memory = data_in.data();
    size_t memory_size = data_in.size();

    // Start timer
    auto start = std::chrono::steady_clock::now();

    // Run the program
    for (int i = 0; i < repeat_N; i++) {
        prog.bpftime_prog_exec(memory, memory_size, &return_val);
    }

    // End timer
    auto end = std::chrono::steady_clock::now();

    // Calculate average time in ns
    auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / repeat_N;

    std::cout << "Time taken: " << time_ns << " ns" << std::endl;
    std::cout << "Return value: " << return_val << std::endl;

    return 0;
}

int bpftime_run_ebpf_program(int id,
			     const std::string &data_in_file, int repeat_N, const std::string &run_type)
{
	cout << "Running eBPF program with id " << id << " and data in file " << data_in_file << endl;
	cout << "Repeat N: " << repeat_N << " with run type " << run_type << endl;
	std::vector<uint8_t> data_in;
	std::ifstream ifs(data_in_file, std::ios::binary | std::ios::ate);
	if (!ifs.is_open()) {
		cerr << "Unable to open data in file" << endl;
		return 1;
	}
	std::streamsize size = ifs.tellg();
	ifs.seekg(0, std::ios::beg);
	data_in.resize(size);
	if (!ifs.read((char *)data_in.data(), size)) {
		cerr << "Unable to read data in file" << endl;
		return 1;
	}
	bpftime_initialize_global_shm(shm_open_type::SHM_OPEN_ONLY);
	const handler_manager *manager =
		shm_holder.global_shared_memory.get_manager();
	size_t handler_size = manager->size();
	if ((size_t) id >= handler_size || id < 0) {
		cerr << "Invalid id " << id << " not exist" << endl;
		return 1;
	}
	if (std::holds_alternative<bpf_prog_handler>(
			    manager->get_handler(id))) {
			const auto &prog = std::get<bpf_prog_handler>(
				manager->get_handler(id));
			auto new_prog = bpftime_prog(prog.insns.data(),
							 prog.insns.size(),
							 prog.name.c_str());
							bpftime::bpftime_helper_group::get_kernel_utils_helper_group()
			.add_helper_group_to_prog(&new_prog);
			bpftime::bpftime_helper_group::get_shm_maps_helper_group()
				.add_helper_group_to_prog(&new_prog);
			if (run_type == "JIT") {
				new_prog.bpftime_prog_load(true);
			} else if (run_type == "AOT") {
				if (prog.aot_insns.size() == 0) {
					cerr << "AOT instructions not found" << endl;
					return 1;
				}
				new_prog.load_aot_object(std::vector<uint8_t>(prog.aot_insns.begin(), prog.aot_insns.end()));
			} else if (run_type == "INTERPRET") {
				new_prog.bpftime_prog_load(false);
			}
			return run_ebpf_and_measure(new_prog, data_in, repeat_N);
	} else {
		cerr << "Invalid id " << id << " not a bpf program" << endl;
		return 1;
	}
	return 0;
}

inline void check_run_type(const std::string &run_type) {
	if (run_type != "JIT" && run_type != "AOT" && run_type != "INTERPRET") {
		cerr << "Invalid run type " << run_type << endl;
		cerr << "Valid run types are JIT, AOT, INTERPRET" << endl;
		exit(1);
	}
}

// Main program
int main(int argc, char *argv[])
{
	if (argc == 1) {
		cerr << "Usage: " << argv[0]
		     << " [load|import|export|remove|run] ..." << endl
		     << "Command-line tool to inspect and manage userspace eBPF objects"
		     << endl;
		return 1;
	}

	spdlog::cfg::load_env_levels();

	auto cmd = std::string(argv[1]);
	if (cmd == "load") {
		if (argc != 3) {
			cerr << "Usage: " << argv[0] << " load <fd> <JSON>"
			     << endl
			     << "Load a JSON file containing eBPF objects into the global shared memory"
			     << endl;
			return 1;
		}
		bpftime_initialize_global_shm(
			shm_open_type::SHM_CREATE_OR_OPEN);
		int fd = atoi(argv[2]);
		auto json_str = std::string(argv[3]);
		return bpftime_import_shm_handler_from_json(fd,
							    json_str.c_str());
	} else if (cmd == "export") {
		if (argc != 3) {
			cerr << "Usage: " << argv[0] << " export <filename>"
			     << endl
			     << "Export the global shared memory to a JSON file"
			     << endl;
			return 1;
		}
		bpftime_initialize_global_shm(shm_open_type::SHM_OPEN_ONLY);
		auto filename = std::string(argv[2]);
		return bpftime_export_global_shm_to_json(filename.c_str());
	} else if (cmd == "import") {
		if (argc != 3) {
			cerr << "Usage: " << argv[0] << " import <filename>"
			     << endl
			     << "Import a JSON file containing eBPF objects into the global shared memory"
			     << endl;
			return 1;
		}
		bpftime_initialize_global_shm(
			shm_open_type::SHM_CREATE_OR_OPEN);
		auto agent_config = get_agent_config_from_env();
		bpftime_set_agent_config(agent_config);
		auto filename = std::string(argv[2]);
		return bpftime_import_global_shm_from_json(filename.c_str());
	} else if (cmd == "remove") {
		if (argc != 2) {
			cerr << "Usage: " << argv[0] << " remove" << endl
			     << "Remove the global shared memory system wide"
			     << endl;
			return 1;
		}
		bpftime_remove_global_shm();
		return 0;
	} else if (cmd == "run") {
		if (argc < 4 || argc > 8) {
			cerr << "Usage: " << argv[0] << " run <id> <data_in_file> [repeat N] [type RUN_TYPE]"
			     << endl
			     << "Run an eBPF program from id with data in from data_in_file"
			     << endl
				 << "RUN_TYPE := AOT | JIT | INTERPRET"
			     << endl;
			return 1;
		}
		int id = atoi(argv[2]);
		auto data_in_file = std::string(argv[3]);
		int repeat_N = 1;
		std::string run_type = "JIT";
		if (argc > 5) {
			if (strcmp(argv[4], "repeat") == 0) {
				repeat_N = atoi(argv[5]);
			} else if (strcmp(argv[4], "type") == 0) {
				run_type = std::string(argv[5]);
			} else {
				cerr << "Invalid subcommand " << argv[4] << endl;
				return 1;
			}
			if (argc > 7) {
 				if (strcmp(argv[6], "type") == 0) {
					run_type = std::string(argv[7]);
				} else {
					cerr << "Invalid subcommand " << argv[6] << endl;
					return 1;
				}
			}
		}
		check_run_type(run_type);
		return bpftime_run_ebpf_program(id, data_in_file, repeat_N, run_type);
	}
	else {
		cerr << "Invalid subcommand " << cmd << endl;
		return 1;
	}
	return EXIT_SUCCESS;
}
