#include "arguments.hpp"
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
#include <thread>
#include <bpftime_logger.hpp>
#include <boost/interprocess/exceptions.hpp>
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
	try {
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
	} catch (const boost::interprocess::interprocess_exception &e) {
		cerr << "Error: Failed to access shared memory: " << e.what() << endl
		     << "Hint: Shared memory may not exist, be corrupted, or permission denied." << endl;
		std::string shm_name = bpftime::get_global_shm_name();
		std::string shm_path = "/dev/shm/" + shm_name;
		cerr << "Current shared memory: " << shm_path << endl;
		if (std::string(e.what()).find("Permission denied") != std::string::npos) {
			cerr << "Try running with sudo or check if the shared memory was created by root." << endl;
		} else {
			cerr << "Hint: Shared memory does not exist. Please ensure bpftime server is running." << endl;
		}
		return 1;
	}
	return 0;
}

// Main program
int main(int argc, char *argv[])
{
	// Resolve the log destination before anything else logs: until this
	// runs, SPDLOG_* goes to spdlog's default logger, which writes to
	// stdout regardless of BPFTIME_LOG_OUTPUT.
	const char *logger_target = getenv("BPFTIME_LOG_OUTPUT");
	bpftime::bpftime_set_logger(logger_target == nullptr ?
					    DEFAULT_LOGGER_OUTPUT_PATH :
					    logger_target);
	spdlog::cfg::load_env_levels();
#if defined(BPFTIME_ENABLE_CUDA_ATTACH)
	tool_arguments args(argv[0], true);
#else
	tool_arguments args(argv[0]);
#endif
	try {
		args.parse(argc, argv);
	} catch (const std::exception &err) {
		cerr << "Error: " << err.what() << endl << args.program;
		return 1;
	}
	if (args.program.is_subcommand_used(args.load)) {
		bpftime_initialize_global_shm(
			shm_open_type::SHM_CREATE_OR_OPEN);
		int fd = args.load.get<int>("fd");
		auto json_str = args.load.get<std::string>("JSON");
		return bpftime_import_shm_handler_from_json(fd,
							    json_str.c_str());
	} else if (args.program.is_subcommand_used(args.export_cmd)) {
		auto filename = args.export_cmd.get<std::string>("filename");

		// Check if output file is the current shared memory device
		std::string shm_name = bpftime::get_global_shm_name();
		std::string shm_path = "/dev/shm/" + shm_name;
		if (filename == shm_path) {
			cerr << "Error: Output file cannot be the shared memory device file." << endl
			     << "Hint: export outputs to JSON file, not to shared memory itself." << endl
			     << "Current shared memory: " << shm_path << endl
			     << "Example: " << argv[0] << " export /tmp/output.json" << endl;
			return 1;
		}

		try {
			bpftime_initialize_global_shm(shm_open_type::SHM_OPEN_ONLY);
			return bpftime_export_global_shm_to_json(filename.c_str());
		} catch (const boost::interprocess::interprocess_exception &e) {
			cerr << "Error: Failed to access shared memory: " << e.what() << endl
			     << "Hint: Shared memory may not exist, be corrupted, or permission denied." << endl
			     << "Current shared memory: " << shm_path << endl;
			if (std::string(e.what()).find("Permission denied") != std::string::npos) {
				cerr << "Try running with sudo or check if the shared memory was created by root." << endl;
			} else {
				cerr << "Try:" << endl
				     << "  1. Remove corrupted shared memory: rm -f " << shm_path << endl
				     << "  2. Restart bpftime server: bpftime load <your_ebpf_program>" << endl
				     << "  3. Or use: " << argv[0] << " remove" << endl;
			}
			return 1;
		}
	} else if (args.program.is_subcommand_used(args.import_cmd)) {
		bpftime_initialize_global_shm(
			shm_open_type::SHM_CREATE_OR_OPEN);
		auto runtime_config = bpftime::construct_runtime_config_from_env();
		bpftime_set_runtime_config(std::move(runtime_config));
		auto filename = args.import_cmd.get<std::string>("filename");
		return bpftime_import_global_shm_from_json(filename.c_str());
	} else if (args.program.is_subcommand_used(args.remove)) {
		bpftime_remove_global_shm();
		return 0;
	} else if (args.program.is_subcommand_used(args.run)) {
		return bpftime_run_ebpf_program(
			args.run.get<int>("id"), args.run.get<std::string>("data_file"),
			args.run.is_used("--repeat") ? args.run.get<int>("--repeat") : 1,
			args.run.is_used("--type") ? args.run.get<std::string>("--type") : "JIT");
	}
#if defined(BPFTIME_ENABLE_CUDA_ATTACH)
	else if (args.program.is_subcommand_used(args.cuda)) {
		auto program_name = args.cuda.get<std::string>("program");
		int run_count = args.cuda.present<int>("count").value_or(1);
		auto dims = args.cuda.get<std::vector<int>>("dimensions");
		try {
			bpftime_initialize_global_shm(shm_open_type::SHM_OPEN_ONLY);
			auto &runtime_config = bpftime_get_runtime_config();
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
				const int id = (*impl)->find_attach_entry_by_program_name(
					program_name.c_str());
				if (id == -1) {
					SPDLOG_ERROR("Unable to find program: {}",
						     program_name.c_str());
					return 1;
				}
				int err = 0;
				if (!dims.empty()) {
					err = (*impl)->run_attach_entry_on_gpu(
						id, run_count, dims[0], dims[1], dims[2],
						dims[3], dims[4], dims[5]);
				} else {
					err = (*impl)->run_attach_entry_on_gpu(
						id, run_count);
				}
				if (err) {
					SPDLOG_ERROR("Unable to run program: {}", err);
					return 1;
				}
				return 0;
			}
			SPDLOG_ERROR("nv_attach_impl not found!");
			return 1;
		} catch (const boost::interprocess::interprocess_exception &e) {
			cerr << "Error: Failed to access shared memory: " << e.what() << endl
			     << "Hint: Shared memory may not exist, be corrupted, or permission denied." << endl;
			std::string shm_name = bpftime::get_global_shm_name();
			std::string shm_path = "/dev/shm/" + shm_name;
			cerr << "Current shared memory: " << shm_path << endl;
			if (std::string(e.what()).find("Permission denied") != std::string::npos) {
				cerr << "Try running with sudo or check if the shared memory was created by root." << endl;
			} else {
				cerr << "Hint: Shared memory does not exist. Please ensure bpftime server is running." << endl;
			}
			return 1;
		}
	}
#endif
	return EXIT_SUCCESS;
}
