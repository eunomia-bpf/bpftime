#pragma once

#include <argparse/argparse.hpp>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <string>
#include <vector>

// Own subparsers for as long as the root parser references them. Keeping the
// command schema independent of the runtime also permits CUDA argument tests
// on machines without CUDA.
struct tool_arguments {
	argparse::ArgumentParser program;
	argparse::ArgumentParser load{"load", "", argparse::default_arguments::help};
	argparse::ArgumentParser export_cmd{"export", "", argparse::default_arguments::help};
	argparse::ArgumentParser import_cmd{"import", "", argparse::default_arguments::help};
	argparse::ArgumentParser remove{"remove", "", argparse::default_arguments::help};
	argparse::ArgumentParser run{"run", "", argparse::default_arguments::help};
	argparse::ArgumentParser cuda{"run-on-cuda", "", argparse::default_arguments::help};

	static auto integer(const std::string &name, int minimum)
	{
		// Retain decimal +N/leading whitespace compatibility with strtol.
		return [name, minimum](const std::string &value) {
			errno = 0;
			char *end = nullptr;
			long number = strtol(value.c_str(), &end, 10);
			if (errno != 0 || end == value.c_str() || *end != '\0' ||
			    number < minimum || number > INT_MAX) {
				throw std::runtime_error("invalid " + name + " '" + value +
					"'; expected an integer from " +
					std::to_string(minimum) + " to " + std::to_string(INT_MAX));
			}
			return static_cast<int>(number);
		};
	}

	explicit tool_arguments(const char *name, bool enable_cuda = false)
		: program(name, "", argparse::default_arguments::help)
	{
		program.add_description("Inspect and manage userspace eBPF objects");
		load.add_argument("fd").action(integer("fd", 0));
		load.add_argument("JSON");
		export_cmd.add_argument("filename");
		import_cmd.add_argument("filename");
		run.add_argument("id").action(integer("id", 0));
		run.add_argument("data_file");
		// This argparse version accepts a missing option value if it has
		// default_value(). Apply defaults only after successful parsing.
		run.add_argument("--repeat").action(integer("repeat", 1));
		run.add_argument("--type").choices("JIT", "AOT", "INTERPRET");
		run.add_epilog("Legacy syntax: run <id> <data_file> [repeat N] [type RUN_TYPE]");
		program.add_subparser(load);
		program.add_subparser(export_cmd);
		program.add_subparser(import_cmd);
		program.add_subparser(remove);
		program.add_subparser(run);
		if (enable_cuda) {
			cuda.add_argument("program");
			cuda.add_argument("count").nargs(0, 1).action(integer("count", 1));
			cuda.add_argument("dimensions").nargs(0, 6)
				.action(integer("dimension", 1))
				.help("gridX gridY gridZ blockX blockY blockZ (all six required)");
			program.add_subparser(cuda);
		}
	}

	void parse(int argc, char **argv)
	{
		std::vector<std::string> args(argv, argv + argc);
		if (argc > 1 && args[1] == "run") {
			size_t positionals = 0;
			for (size_t i = 2; i < args.size(); ++i) {
				auto &arg = args[i];
				// Canonical options may precede the positionals. Skip
				// their values so filenames "type"/"repeat" stay literal.
				if (positionals >= 2 && (arg == "repeat" || arg == "type"))
					arg = "--" + arg;
				if (arg == "--repeat" || arg == "--type") {
					++i;
				} else if (arg.rfind("--", 0) != 0) {
					++positionals;
				}
			}
		}
		program.parse_args(args);
		if (argc == 1)
			throw std::runtime_error("a command is required");
		if (argc > 1 && args[1] == "run-on-cuda") {
			auto dims = cuda.get<std::vector<int>>("dimensions");
			if (!dims.empty() && dims.size() != 6)
				throw std::runtime_error("provide all six CUDA dimensions");
		}
	}
};
