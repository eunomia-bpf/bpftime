#include "spdlog/spdlog.h"
#include <cstdlib>
#include <frida-core.h>
#include <argparse/argparse.hpp>
#include <filesystem>
#include <stdexcept>
#include <unistd.h>
#include <vector>
#include <string>
#include <utility>
#include <tuple>
#include <sys/wait.h>
static int run_command(const char *path, const std::vector<std::string> &argv,
		       const char *ld_preload, const char *agent_so)
{
	int pid = fork();
	if (pid == 0) {
		std::string ld_preload_str("LD_PRELOAD=");
		std::string agent_so_str("AGENT_SO=");
		ld_preload_str += ld_preload;
		const char *env[] = { ld_preload_str.c_str(), nullptr,
				      nullptr };
		if (agent_so) {
			agent_so_str += agent_so;
			env[1] = agent_so_str.c_str();
		}
		std::vector<const char *> argv_arr;
		for (const auto &str : argv)
			argv_arr.push_back(str.c_str());
		argv_arr.push_back(nullptr);
		execvpe(path, (char *const *)argv_arr.data(),
			(char *const *)env);
	} else {
		int status;
		if (int cid = waitpid(pid, &status, 0); cid > 0) {
			if (WIFEXITED(status)) {
				int exit_code = WEXITSTATUS(status);
				if (exit_code != 0) {
					spdlog::error(
						"Program exited abnormally: {}",
						exit_code);
					return 1;
				}
			}
		}
	}
	return 1;
}
static int inject_by_frida(int pid, const char *inject_path, const char *arg)
{
	frida_init();
	auto injector = frida_injector_new();
	GError *err = nullptr;
	auto id = frida_injector_inject_library_file_sync(injector, pid,
							  inject_path,
							  "bpftime_agent_main",
							  arg, nullptr, &err);
	if (err) {
		spdlog::error("Failed to inject: {}", err->message);
		g_error_free(err);
		frida_unref(injector);
		frida_deinit();
		return 1;
	}
	spdlog::info("Successfully injected. ID: {}", id);
	frida_injector_close_sync(injector, nullptr, nullptr);
	frida_unref(injector);
	frida_deinit();
	return 0;
}

static std::pair<std::string, std::vector<std::string> >
extract_path_and_args(const argparse::ArgumentParser &parser)
{
	std::vector<std::string> args;
	try {
		args = parser.get<std::vector<std::string> >("EXTRA_ARGS");
	} catch (std::logic_error &e) {
	}
	for (auto s : args)
		spdlog::info("{}", s);
	return std::make_pair(parser.get("EXECUTABLE_PATH").c_str(), args);
}
int main(int argc, const char **argv)
{
	argparse::ArgumentParser program(argv[0]);

	if (auto home_env = getenv("HOME"); home_env) {
		std::string default_location(home_env);
		default_location += "/.bpftime";
		program.add_argument("-i", "--install-location")
			.help("Installing location of bpftime")
			.default_value(default_location)
			.required()
			.nargs(1);
	} else {
		spdlog::warn(
			"Unable to determine home directory. You must specify --install-location");
		program.add_argument("-i", "--install-location")
			.help("Installing location of bpftime")
			.required()
			.nargs(1);
	}

	program.add_argument("-d", "--dry-run")
		.help("Run without commiting any modifications")
		.flag();

	argparse::ArgumentParser load_command(
		"load", "1.0", argparse::default_arguments::none);
	load_command.add_argument("-h", "--help")
		.action([&](const std::string &s) {
			std::cout << load_command.help().str();
			exit(0);
		})
		.default_value(false)
		.help("shows help message")
		.implicit_value(true)
		.nargs(0);
	load_command.add_description(
		"Start an application with bpftime-server injected");
	load_command.add_argument("EXECUTABLE_PATH")
		.help("Path to the executable that will be injected with syscall-server");
	load_command.add_argument("EXTRA_ARGS")
		.help("Other arguments to the program injected")
		.remaining();

	argparse::ArgumentParser start_command(
		"start", "1.0", argparse::default_arguments::none);
	start_command.add_argument("-h", "--help")
		.action([&](const std::string &s) {
			std::cout << start_command.help().str();
		})
		.default_value(false)
		.help("shows help message")
		.implicit_value(true)
		.nargs(0);

	start_command.add_description(
		"Start an application with bpftime-agent injected");
	start_command.add_argument("-s", "--enable-syscall-trace")
		.help("Whether to enable syscall trace")
		.flag();
	start_command.add_argument("EXECUTABLE_PATH")
		.help("Path to the executable that will be injected with agent");
	start_command.add_argument("EXTRA_ARGS")
		.help("Other arguments to the program injected")
		
		// .scan<char Shape, typename T>()
		.remaining();

	argparse::ArgumentParser attach_command(
		"attach", "1.0", argparse::default_arguments::none);
	// attach_command.add_argument("-h", "--help")
	// 	.action([&](const std::string &s) {
	// 		std::cout << attach_command.help().str();
	// 	})
	// 	.default_value(false)
	// 	.help("shows help message")
	// 	.implicit_value(true)
	// 	.nargs(0);
	attach_command.add_description("Inject bpftime-agent to a certain pid");
	attach_command.add_argument("-s", "--enable-syscall-trace")
		.help("Whether to enable syscall trace")
		.flag();
	attach_command.add_argument("PID").scan<'i', int>();

	program.add_subparser(load_command);
	program.add_subparser(start_command);
	program.add_subparser(attach_command);
	// program.set_suppress(false);
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
	std::filesystem::path install_path(program.get("install-location"));
	if (program.is_subcommand_used("load")) {
		auto so_path = install_path / "libbpftime-syscall-server.so";
		if (!std::filesystem::exists(so_path)) {
			spdlog::error("Library not found: {}", so_path.c_str());
			return 1;
		}
		auto [executable_path, extra_args] =
			extract_path_and_args(load_command);
		return run_command(executable_path.c_str(), extra_args,
				   so_path.c_str(), nullptr);
	} else if (program.is_subcommand_used("start")) {
		auto agent_path = install_path / "libbpftime-syscall-server.so";
		if (!std::filesystem::exists(agent_path)) {
			spdlog::error("Library not found: {}",
				      agent_path.c_str());
			return 1;
		}
		auto [executable_path, extra_args] =
			extract_path_and_args(start_command);
		if (start_command.get<bool>("enable-syscall-trace")) {
			auto transformer_path =
				install_path /
				"libbpftime-agent-transformer.so";
			if (!std::filesystem::exists(transformer_path)) {
				spdlog::error("Library not found: {}",
					      transformer_path.c_str());
				return 1;
			}

			return run_command(executable_path.c_str(), extra_args,
					   transformer_path.c_str(),
					   agent_path.c_str());
		} else {
			return run_command(executable_path.c_str(), extra_args,
					   agent_path.c_str(), nullptr);
		}
	} else if (program.is_subcommand_used("attach")) {
		auto agent_path = install_path / "libbpftime-syscall-server.so";
		if (!std::filesystem::exists(agent_path)) {
			spdlog::error("Library not found: {}",
				      agent_path.c_str());
			return 1;
		}
		auto pid = attach_command.get<int>("PID");
		if (attach_command.get<bool>("enable-syscall-trace")) {
			auto transformer_path =
				install_path /
				"libbpftime-agent-transformer.so";
			if (!std::filesystem::exists(transformer_path)) {
				spdlog::error("Library not found: {}",
					      transformer_path.c_str());
				return 1;
			}
			return inject_by_frida(pid, transformer_path.c_str(),
					       agent_path.c_str());
		} else {
			return inject_by_frida(pid, agent_path.c_str(), "");
		}
	}
	return 0;
}
