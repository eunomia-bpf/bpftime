#include "ebpf-vm.h"
#include "llvm_bpf_jit.h"
#include "bpftime_helper_group.hpp"
#include "bpftime_prog.hpp"
#include "bpftime_shm.hpp"
#include "bpf_attach_ctx.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/cfg/env.h"
#include <argparse/argparse.hpp>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <libelf.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/llvm_jit_context.hpp>
#include <string>
#include <unistd.h>
#include <bpf/libbpf.h>
#include <fstream>
#include <bpftime_shm_internal.hpp>
#include <cassert>
#include <cstddef>
#include <variant>
#include <vector>

using namespace llvm;
using namespace llvm::orc;

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

union bpf_attach_ctx_holder {
	bpftime::bpf_attach_ctx ctx;
	bpf_attach_ctx_holder()
	{
	}
	~bpf_attach_ctx_holder()
	{
	}
	void destroy()
	{
		ctx.~bpf_attach_ctx();
	}
	void init()
	{
		new (&ctx) bpftime::bpf_attach_ctx;
	}
};

static bpf_attach_ctx_holder ctx_holder;

bool emit_llvm_ir = false;

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
		bpftime::bpftime_prog bpftime_prog((const ebpf_inst *)	bpf_program__insns(prog),
					   bpf_program__insn_cnt(prog),
					   name);
		bpftime::bpftime_helper_group::get_kernel_utils_helper_group()
			.add_helper_group_to_prog(&bpftime_prog);
		bpftime::bpftime_helper_group::get_shm_maps_helper_group()
			.add_helper_group_to_prog(&bpftime_prog);
		bpftime_prog.bpftime_prog_load(true);
		llvm_bpf_jit_context ctx(bpftime_prog.get_vm());
		auto result = ctx.do_aot_compile(emit_llvm_ir);
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
	ctx_holder.init();
	const handler_manager *manager =
		shm_holder.global_shared_memory.get_manager();
	size_t handler_size = manager->size();
	// TODO: fix load programs
	for (size_t i = 0; i < manager->size(); i++) {
		if (std::holds_alternative<bpf_prog_handler>(
			    manager->get_handler(i))) {
			const auto &prog = std::get<bpf_prog_handler>(
				manager->get_handler(i));
			// temp work around: we need to create new attach points
			// in the runtime
			// TODO: fix this hard code name
			auto new_prog = bpftime_prog(prog.insns.data(),
							 prog.insns.size(),
							 prog.name.c_str());
					bpftime::bpftime_helper_group::get_kernel_utils_helper_group()
			.add_helper_group_to_prog(&new_prog);
			bpftime::bpftime_helper_group::get_shm_maps_helper_group()
				.add_helper_group_to_prog(&new_prog);
			new_prog.bpftime_prog_load(true);
			llvm_bpf_jit_context ctx(new_prog.get_vm());
			auto result = ctx.do_aot_compile(emit_llvm_ir);
			auto out_path = output / (std::string(prog.name.c_str()) + ".o");
			std::ofstream ofs(out_path, std::ios::binary);
			ofs.write((const char *)result.data(), result.size());
			return 0;
		}
	}
	return 0;
}

static ExitOnError exit_on_error;
using bpf_func = uint64_t (*)(const void *, uint64_t);

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
	auto vm = bpftime_prog.get_vm();
	vm->jit_context->load_aot_object(file_buf);
	int ret = vm->jit_context->get_entry_address()(&mem, sizeof(mem));
	SPDLOG_INFO("Output: {}", ret);
	return 0;
}

int main(int argc, const char **argv)
{
	spdlog::cfg::load_env_levels();
	libbpf_set_print(_libbpf_print);
	InitializeNativeTarget();
	InitializeNativeTargetAsmPrinter();
	argparse::ArgumentParser program(argv[0]);

	argparse::ArgumentParser build_command("build");
	build_command.add_description(
		"Build native ELF(s) from eBPF ELF. Each program in the eBPF ELF will be built into a single native ELF");
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
	compile_command.add_description("Compile the eBPF program loaded in shared memory");
	compile_command.add_argument("-o", "--output")
		.default_value(".")
		.help("Output directory (There might be multiple output files for a single input)");
	compile_command.add_argument("-e", "--emit_llvm")
		.default_value(false)
		.implicit_value(true)
		.help("Emit LLVM IR for the eBPF program");

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
		return compile_ebpf_program(compile_command.get<std::string>("output"));
	}
	return 0;
}
