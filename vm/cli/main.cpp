#include <argparse/argparse.hpp>
#include <llvm/llvm_jit_context.hpp>

int main(int argc, const char **argv)
{
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
	} else if (program.is_subcommand_used(run_command)) {
	}
	return 0;
}
