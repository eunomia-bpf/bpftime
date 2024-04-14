#include <compat_llvm.hpp>
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
		bpftime::vm::llvm::bpftime_llvm_jit_vm vm;

		if (vm.load_code((const void *)bpf_program__insns(prog),
				 (uint32_t)bpf_program__insn_cnt(prog) * 8) <
		    0) {
			SPDLOG_ERROR(
				"Unable to load instruction of program {}: ",
				name, vm.get_error_message());
			return 1;
		}
		llvm_bpf_jit_context ctx(&vm);
		auto result = ctx.do_aot_compile();
		auto out_path = output / (std::string(name) + ".o");
		std::ofstream ofs(out_path, std::ios::binary);
		ofs.write((const char *)result.data(), result.size());
		SPDLOG_INFO("Program {} written to {}", name, out_path.c_str());
	}
	return 0;
}

static ExitOnError exit_on_error;
using bpf_func = uint64_t (*)(const void *, uint64_t);

static int run_ebpf_program(const std::filesystem::path &elf,
			    std::optional<std::string> memory)
{
	auto jit = exit_on_error(LLJITBuilder().create());

	if (auto file_buf = MemoryBuffer::getFile(elf.c_str()); file_buf) {
		exit_on_error(jit->addObjectFile(std::move(*file_buf)));
	} else {
		SPDLOG_ERROR("Unable to read elf file: {}",
			     file_buf.getError().message());
		return 1;
	}
	if (auto ret = jit->lookup("bpf_main"); ret) {
		auto func = ret->toPtr<bpf_func>();
		uint64_t result;
		if (memory.has_value()) {
			std::ifstream ifs(*memory,
					  std::ios::binary | std::ios::ate);
			if (!ifs.is_open()) {
				SPDLOG_ERROR("Unable to open memory file");
				return 1;
			}
			std::streamsize size = ifs.tellg();
			ifs.seekg(0, std::ios::beg);
			std::vector<uint8_t> buffer(size);
			if (!ifs.read((char *)buffer.data(), size)) {
				SPDLOG_ERROR("Unable to read memory");
				return 1;
			}
			SPDLOG_INFO("Memory size: {}", size);
			result = func(buffer.data(), buffer.size());
		} else {
			result = func(nullptr, 0);
		}
		SPDLOG_INFO("Output: {}", result);
		return 0;
	} else {
		std::string buf;
		raw_string_ostream os(buf);
		os << ret.takeError();
		SPDLOG_ERROR("Unable to lookup bpf_main: {}", buf);
		return 1;
	}
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

	argparse::ArgumentParser run_command("run");
	run_command.add_description("Run an native eBPF program");
	run_command.add_argument("PATH").help("Path to the ELF file");
	run_command.add_argument("MEMORY")
		.help("Path to the memory file")
		.nargs(0, 1);
	program.add_subparser(build_command);
	program.add_subparser(run_command);

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
	}
	return 0;
}
