#include "ebpf-vm.h"
#include "llvm_bpf_jit.h"
#include "spdlog/spdlog.h"
#include <argparse/argparse.hpp>
#include <cstdarg>
#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <libelf.h>
#include <llvm/llvm_jit_context.hpp>
#include <string>
#include <unistd.h>
#include <bpf/libbpf.h>
#include <fstream>
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
		std::unique_ptr<ebpf_vm, decltype(&ebpf_destroy)> vm(
			ebpf_create(), ebpf_destroy);
		char *errmsg = nullptr;
		int err = ebpf_load(vm.get(),
				    (const void *)bpf_program__insns(prog),
				    (uint32_t)bpf_program__insn_cnt(prog) * 8,
				    &errmsg);
		if (err < 0) {
			SPDLOG_ERROR(
				"Unable to load instruction of program {}: ",
				name, errmsg);
			return 1;
		}
		llvm_bpf_jit_context ctx(vm.get());
		auto result = ctx.do_aot_compile();
		auto out_path = output / (std::string(name) + ".o");
		std::ofstream ofs(out_path, std::ios::binary);
		ofs.write((const char *)result.data(), result.size());
		SPDLOG_INFO("Program {} written to {}", name, out_path.c_str());
	}
	return 0;
}

static int run_ebpf_program(const std::filesystem::path &elf)
{
	return 0;
}

int main(int argc, const char **argv)
{
	libbpf_set_print(_libbpf_print);

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
		return run_ebpf_program(run_command.get<std::string>("PATH"));
	}
	return 0;
}
