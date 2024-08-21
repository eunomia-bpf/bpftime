#include "ebpf-vm.h"
#include "bpftime_helper_group.hpp"
#include "bpftime_prog.hpp"
#include "bpftime_shm.hpp"
#include "bpf_attach_ctx.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/cfg/env.h"
#include <argparse/argparse.hpp>
#include <cstdarg>
#include <cstdint>
#include <cassert>
#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <libelf.h>
#include <string>
#include <unistd.h>
#include <bpf/libbpf.h>
#include <fstream>
#include <bpftime_shm_internal.hpp>
#include <cassert>
#include <cstddef>
#include <variant>
#include <vector>
#include <bpftime_vm_compat.hpp>

extern "C" int _libbpf_print(libbpf_print_level level, const char *fmt,
			     va_list ap)
{
	char buf[2048];
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
	int ret = vsnprintf(buf, sizeof(buf), fmt, ap);
#pragma GCC diagnostic pop
	std::string out(buf);
	if (out.back() == '\n')
		out.pop_back();
	if (level == LIBBPF_WARN) {
		SPDLOG_WARN("{}", out);
	} else if (level == LIBBPF_INFO) {
		SPDLOG_INFO("{}", out);
	} else if (level == LIBBPF_DEBUG) {
		SPDLOG_INFO("{}", out);
	}
	return ret;
}

bool emit_llvm_ir = false;

int empty_helper()
{
	std::cerr << "Empty helper called" << std::endl;
	return 0;
}

bpftime::bpftime_helper_group create_all_helpers()
{
	bpftime::bpftime_helper_group helper_group;
	for (int i = 0; i < 1000; i++) {
		// add empty helpers so that we can be compatible with other use
		// cases
		bpftime::bpftime_helper_info info;
		info.index = i;
		info.name = "empty_helper";
		info.fn = (void *)empty_helper;
		helper_group.register_helper(info);
	}
	return helper_group;
}

static int build_ebpf_program(const std::string &ebpf_elf,
			      const std::filesystem::path &output)
{
	bpf_object *obj = bpf_object__open(ebpf_elf.c_str());
	if (!obj) {
		SPDLOG_CRITICAL("Unable to open BPF elf: {}", errno);
		return 1;
	}
	std::unique_ptr<bpf_object, decltype(&bpf_object__close)> elf(
		obj, bpf_object__close);
	bpf_program *prog;
	bpf_object__for_each_program(prog, elf.get())
	{
		auto name = bpf_program__name(prog);
		SPDLOG_INFO("Processing program {}", name);
		bpftime::bpftime_prog bpftime_prog(
			(const ebpf_inst *)bpf_program__insns(prog),
			bpf_program__insn_cnt(prog), name);
		auto helper_group = create_all_helpers();
		helper_group.add_helper_group_to_prog(&bpftime_prog);
		bpftime_prog.bpftime_prog_load(true);
		auto vm = bpftime_prog.get_vm();
		// The vm instance should not be empty
		assert(vm && vm->vm_instance);
		auto result = vm->vm_instance->do_aot_compile(emit_llvm_ir);
		auto out_path = output / (std::string(name) + ".o");
		std::ofstream ofs(out_path, std::ios::binary);
		ofs.write((const char *)result.data(), result.size());
		SPDLOG_INFO("Program {} written to {}", name, out_path.c_str());
	}
	return 0;
}

static int compile_ebpf_program(const std::filesystem::path &output)
{
	using namespace bpftime;
	bpftime_initialize_global_shm(shm_open_type::SHM_OPEN_ONLY);
	const handler_manager *manager =
		shm_holder.global_shared_memory.get_manager();
	size_t handler_size = manager->size();
	for (size_t i = 0; i < manager->size(); i++) {
		if (std::holds_alternative<bpf_prog_handler>(
			    manager->get_handler(i))) {
			bpf_prog_handler &prog =
				(bpf_prog_handler &)std::get<bpf_prog_handler>(
					manager->get_handler(i));
			auto new_prog = bpftime_prog(prog.insns.data(),
						     prog.insns.size(),
						     prog.name.c_str());
			auto helper_group = create_all_helpers();
			helper_group.add_helper_group_to_prog(&new_prog);
			new_prog.bpftime_prog_load(true);
			auto vm = new_prog.get_vm();
			// The vm instance should not be empty
			assert(vm && vm->vm_instance);
			auto result =
				vm->vm_instance->do_aot_compile(emit_llvm_ir);
			auto out_path = output /
					(std::string(prog.name.c_str()) + ".o");
			std::ofstream ofs(out_path, std::ios::binary);
			ofs.write((const char *)result.data(), result.size());
			// update the aot_insns in share memory
			prog.aot_insns.resize(result.size());
			std::copy(result.begin(), result.end(),
				  prog.aot_insns.begin());
			return 0;
		}
	}
	return 0;
}

static int load_ebpf_program(const std::filesystem::path &elf, int id)
{
	using namespace bpftime;
	// read the file
	std::ifstream ifs(elf, std::ios::binary | std::ios::ate);
	if (!ifs.is_open()) {
		SPDLOG_ERROR("Unable to open ELF file");
		return 1;
	}
	std::streamsize size = ifs.tellg();
	ifs.seekg(0, std::ios::beg);
	std::vector<uint8_t> file_buf(size);
	if (!ifs.read((char *)file_buf.data(), size)) {
		SPDLOG_ERROR("Unable to read ELF");
		return 1;
	}
	bpftime_initialize_global_shm(shm_open_type::SHM_OPEN_ONLY);
	const handler_manager *manager =
		shm_holder.global_shared_memory.get_manager();
	size_t handler_size = manager->size();
	if (id >= (int)handler_size || id < 0) {
		SPDLOG_ERROR("Invalid id {} not exist", id);
		return 1;
	}
	if (std::holds_alternative<bpf_prog_handler>(
		    manager->get_handler(id))) {
		bpf_prog_handler &prog =
			(bpf_prog_handler &)std::get<bpf_prog_handler>(
				manager->get_handler(id));
		// update the aot_insns in share memory
		prog.aot_insns.resize(file_buf.size());
		std::copy(file_buf.begin(), file_buf.end(),
			  prog.aot_insns.begin());
		return 0;
	} else {
		SPDLOG_ERROR("Invalid id {} not a bpf program", id);
		return 1;
	}
	return 0;
}

static int run_ebpf_program(const std::filesystem::path &elf,
			    std::optional<std::string> memory)
{
	// read the file
	std::ifstream ifs(elf, std::ios::binary | std::ios::ate);
	if (!ifs.is_open()) {
		SPDLOG_ERROR("Unable to open ELF file");
		return 1;
	}
	std::streamsize size = ifs.tellg();
	ifs.seekg(0, std::ios::beg);
	std::vector<uint8_t> file_buf(size);
	if (!ifs.read((char *)file_buf.data(), size)) {
		SPDLOG_ERROR("Unable to read ELF");
		return 1;
	}
	std::vector<uint8_t> mem;
	if (memory.has_value()) {
		std::ifstream ifs(*memory, std::ios::binary | std::ios::ate);
		if (!ifs.is_open()) {
			SPDLOG_ERROR("Unable to open memory file");
			return 1;
		}
		std::streamsize size = ifs.tellg();
		ifs.seekg(0, std::ios::beg);
		mem.resize(size);
		if (!ifs.read((char *)mem.data(), size)) {
			SPDLOG_ERROR("Unable to read memory");
			return 1;
		}
		SPDLOG_INFO("Memory size: {}", size);
	}

	const ebpf_inst insn[1] = {};
	bpftime::bpftime_prog bpftime_prog(insn, 0, "bpf_main");
	bpftime::bpftime_helper_group::get_kernel_utils_helper_group()
		.add_helper_group_to_prog(&bpftime_prog);
	bpftime::bpftime_helper_group::get_shm_maps_helper_group()
		.add_helper_group_to_prog(&bpftime_prog);
	bpftime_prog.load_aot_object(file_buf);
	uint64_t retval;
	int ret = bpftime_prog.bpftime_prog_exec(&mem, sizeof(mem), &retval);
	if (ret < 0) {
		SPDLOG_ERROR("Failed to exec the eBPF program: {}", ret);
		return 0;
	}
	SPDLOG_INFO("Output: {}", retval);
	return 0;
}

int main(int argc, const char **argv)
{
	spdlog::cfg::load_env_levels();
	libbpf_set_print(_libbpf_print);
	argparse::ArgumentParser program(argv[0]);

	argparse::ArgumentParser build_command("build");
	build_command.add_description(
		"Build native ELF(s) from eBPF ELF Object."
		"Each program in the eBPF ELF object will be built into a single native ELF");
	build_command.add_argument("-o", "--output")
		.default_value(".")
		.help("Output directory (There might be multiple output files for a single input)");
	build_command.add_argument("EBPF_ELF")
		.help("Path to an eBPF ELF executable");
	build_command.add_argument("-e", "--emit_llvm")
		.default_value(false)
		.implicit_value(true)
		.help("Emit LLVM IR for the eBPF program");

	argparse::ArgumentParser run_command("run");
	run_command.add_description("Run an native eBPF program");
	run_command.add_argument("PATH").help("Path to the ELF file");
	run_command.add_argument("MEMORY")
		.help("Path to the memory file")
		.nargs(0, 1);

	argparse::ArgumentParser compile_command("compile");
	compile_command.add_description(
		"Compile the eBPF program loaded in shared memory");
	compile_command.add_argument("-o", "--output")
		.default_value(".")
		.help("Output directory (There might be multiple output files for a single input)");
	compile_command.add_argument("-e", "--emit_llvm")
		.default_value(false)
		.implicit_value(true)
		.help("Emit LLVM IR for the eBPF program");

	argparse::ArgumentParser load_command("load");
	load_command.add_description(
		"Load an eBPF AOTed ELF file into shared memory");
	load_command.add_argument("PATH").help("Path to the ELF file");
	load_command.add_argument("ID").help("ID of the program to load");

	program.add_subparser(load_command);
	program.add_subparser(build_command);
	program.add_subparser(run_command);
	program.add_subparser(compile_command);

	try {
		program.parse_args(argc, argv);
	} catch (const std::exception &err) {
		std::cerr << err.what() << std::endl;
		std::cerr << program;
		std::exit(1);
	}
	if (!program) {
		std::cerr << program;
		std::exit(1);
	}
	if (program.is_subcommand_used(build_command)) {
		emit_llvm_ir = build_command.get<bool>("emit_llvm");
		return build_ebpf_program(
			build_command.get<std::string>("EBPF_ELF"),
			build_command.get<std::string>("output"));
	} else if (program.is_subcommand_used(run_command)) {
		if (run_command.is_used("MEMORY")) {
			return run_ebpf_program(
				run_command.get<std::string>("PATH"),
				run_command.get<std::string>("MEMORY"));
		} else {
			return run_ebpf_program(
				run_command.get<std::string>("PATH"), {});
		}
	} else if (program.is_subcommand_used(compile_command)) {
		emit_llvm_ir = compile_command.get<bool>("emit_llvm");
		return compile_ebpf_program(
			compile_command.get<std::string>("output"));
	} else if (program.is_subcommand_used(load_command)) {
		auto id_str = load_command.get<std::string>("ID");
		return load_ebpf_program(load_command.get<std::string>("PATH"),
					 atoi(id_str.c_str()));
	}
	return 0;
}
