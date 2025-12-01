#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <memory>
#include <bpftime_shm.hpp>
#include <ostream>
#include <stdlib.h>
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
#include <bpftime_logger.hpp>
#ifdef BPFTIME_ENABLE_CUDA_ATTACH
#include <nv_attach_impl.hpp>
#include <nv_attach_private_data.hpp>
#endif
using namespace std;
using namespace bpftime;

static int run_ebpf_and_measure(bpftime_prog &prog,
				std::vector<uint8_t> &data_in, int repeat_N)
{
	// Test the program
	uint64_t return_val;
	void *memory = data_in.data();
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
	auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
			       end - start)
			       .count() /
		       repeat_N;

	std::cout << "Time taken: " << time_ns << " ns" << std::endl;
	std::cout << "Return value: " << return_val << std::endl;

	return 0;
}

int bpftime_run_ebpf_program(int id, const std::string &data_in_file,
			     int repeat_N, const std::string &run_type)
{
	cout << "Running eBPF program with id " << id << " and data in file "
	     << data_in_file << endl;
	cout << "Repeat N: " << repeat_N << " with run type " << run_type
	     << endl;
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
	if ((size_t)id >= handler_size || id < 0) {
		cerr << "Invalid id " << id << " not exist" << endl;
		return 1;
	}
	if (std::holds_alternative<bpf_prog_handler>(
		    manager->get_handler(id))) {
		const auto &prog =
			std::get<bpf_prog_handler>(manager->get_handler(id));
		auto new_prog =
			bpftime_prog(prog.insns.data(), prog.insns.size(),
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
			new_prog.load_aot_object(std::vector<uint8_t>(
				prog.aot_insns.begin(), prog.aot_insns.end()));
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

inline void check_run_type(const std::string &run_type)
{
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
#if defined(BPFTIME_ENABLE_CUDA_ATTACH)
		cerr << "Usage: " << argv[0]
		     << " [load|import|export|remove|run|run-on-cuda] ..."
		     << endl
		     << "Command-line tool to inspect and manage userspace eBPF objects"
		     << endl;

#else
		cerr << "Usage: " << argv[0]
		     << " [load|import|export|remove|run] ..." << endl
		     << "Command-line tool to inspect and manage userspace eBPF objects"
		     << endl;

#endif
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
		auto agent_config = bpftime::construct_agent_config_from_env();
		bpftime_set_agent_config(std::move(agent_config));
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
			cerr << "Usage: " << argv[0]
			     << " run <id> <data_in_file> [repeat N] [type RUN_TYPE]"
			     << endl
			     << "Run an eBPF program from id with data in from data_in_file"
			     << endl
			     << "RUN_TYPE := AOT | JIT | INTERPRET" << endl;
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
				cerr << "Invalid subcommand " << argv[4]
				     << endl;
				return 1;
			}
			if (argc > 7) {
				if (strcmp(argv[6], "type") == 0) {
					run_type = std::string(argv[7]);
				} else {
					cerr << "Invalid subcommand " << argv[6]
					     << endl;
					return 1;
				}
			}
		}
		check_run_type(run_type);
		return bpftime_run_ebpf_program(id, data_in_file, repeat_N,
						run_type);
	}
#if defined(BPFTIME_ENABLE_CUDA_ATTACH)
    else if (cmd == "run-on-cuda") {
        if (argc != 3 && argc != 4 && argc != 10) {
            cerr << "Usage: " << argv[0]
                 << " run-on-cuda [program name] [run count (optional, defaults to 1)]"
                 << " [gridX gridY gridZ blockX blockY blockZ] (optional)"
                 << endl;
            return 1;
        }
        int run_count = 1;
        if (argc >= 4) {
            run_count = atoi(argv[3]);
        }
        int gridX = 1, gridY = 1, gridZ = 1;
        int blockX = 1, blockY = 1, blockZ = 1;
        bool has_dims = false;
        if (argc == 10) {
            has_dims = true;
            gridX = atoi(argv[4]);
            gridY = atoi(argv[5]);
            gridZ = atoi(argv[6]);
            blockX = atoi(argv[7]);
            blockY = atoi(argv[8]);
            blockZ = atoi(argv[9]);
        }
		bpftime_initialize_global_shm(shm_open_type::SHM_OPEN_ONLY);
		auto &runtime_config = bpftime_get_agent_config();
		bpftime_set_logger(
			std::string(runtime_config.get_logger_output_path()));
		bpf_attach_ctx ctx;
		ctx.register_attach_impl(
			{ bpftime::attach::ATTACH_CUDA_PROBE,
			  bpftime::attach::ATTACH_CUDA_RETPROBE },
			std::make_unique<attach::nv_attach_impl>(),
			[](const std::string_view &sv, int &err) {
				std::unique_ptr<
					bpftime::attach::attach_private_data>
					priv_data = std::make_unique<
						bpftime::attach::
							nv_attach_private_data>();
				if (int e = priv_data->initialize_from_string(
					    sv);
				    e < 0) {
					err = e;
					return std::unique_ptr<
						bpftime::attach::
							attach_private_data>();
				}
				return priv_data;
			});
		ctx.init_attach_ctx_from_handlers(runtime_config);
		if (auto impl = ctx.find_nv_attach_impl(); impl) {
            if (auto id =
                    (*impl)->find_attach_entry_by_program_name(
                        argv[2]);
                id != -1) {
                int err = 0;
                if (has_dims) {
                    err = (*impl)->run_attach_entry_on_gpu(
                        id, run_count, gridX, gridY, gridZ, blockX, blockY, blockZ);
                } else {
                    err = (*impl)->run_attach_entry_on_gpu(
                        id, run_count);
                }
                if (err) {
					SPDLOG_ERROR(
						"Unable to run program: {}",
						err);
					return 1;
				} else {
					return 0;
				}
			} else {
				SPDLOG_ERROR("Unable to find program: {}", id);
				return 1;
			};
		} else {
			SPDLOG_ERROR("nv_attach_impl not found!");
		}
	}
#endif
	else {
		cerr << "Invalid subcommand " << cmd << endl;
		return 1;
	}
	return EXIT_SUCCESS;
}
