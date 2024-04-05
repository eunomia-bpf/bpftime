#include "spdlog/spdlog.h"
#include "spdlog/cfg/env.h"
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <frida-core.h>
#include <argparse/argparse.hpp>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <unistd.h>
#include <vector>
#include <string>
#include <utility>
#include <tuple>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#define STR_MAX 1024

static int subprocess_pid = 0;

static bool str_starts_with(const char *main, const char *pat)
{
	if (strstr(main, pat) == main)
		return true;
	return false;
}

static int run_command(const char *path, const std::vector<std::string> &argv,
		       const char *ld_preload, const char *agent_so,
		       const char *tracepipe_path)
{
	int pid = fork();
	if (pid == 0) {
		std::string ld_preload_str("LD_PRELOAD=");
		std::string agent_so_str("AGENT_SO=");
        std::string tracepipe_path_str("TRACEPIPE_PATH=");
		ld_preload_str += ld_preload;

		if (agent_so) {
			agent_so_str += agent_so;
		}
		if (tracepipe_path) {
            tracepipe_path_str += tracepipe_path;
        }
		std::vector<const char *> env_arr;
		char **p = environ;
		while (*p) {
			env_arr.push_back(*p);
			p++;
		}
		bool ld_preload_set = false, agent_so_set = false,
		     tracepipe_path_set = false;
		for (auto &s : env_arr) {
			if (str_starts_with(s, "LD_PRELOAD=")) {
				s = ld_preload_str.c_str();
				ld_preload_set = true;
			} else if (str_starts_with(s, "AGENT_SO=")) {
				s = agent_so_str.c_str();
				agent_so_set = true;
			} else if (str_starts_with(s, "TRACEPIPE_PATH=")) {
			    s = tracepipe_path_str.c_str();
			    tracepipe_path_set = true;
            }
		}
		if (!ld_preload_set)
			env_arr.push_back(ld_preload_str.c_str());
		if (!agent_so_set)
			env_arr.push_back(agent_so_str.c_str());
		if (!tracepipe_path_set) {
		    env_arr.push_back(tracepipe_path_str.c_str());
        }

		env_arr.push_back(nullptr);
		std::vector<const char *> argv_arr;
		argv_arr.push_back(path);
		for (const auto &str : argv)
			argv_arr.push_back(str.c_str());
		argv_arr.push_back(nullptr);
		execvpe(path, (char *const *)argv_arr.data(),
			(char *const *)env_arr.data());
	} else {
		subprocess_pid = pid;
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
	spdlog::info("Injecting to {}", pid);
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
	std::vector<std::string> items;
	try {
		items = parser.get<std::vector<std::string> >("COMMAND");
	} catch (std::logic_error &err) {
		std::cerr << parser;
		exit(1);
	}
	std::string executable = items[0];
	items.erase(items.begin());
	return { executable, items };
}

static int read_tracepipe(const char *tracepipe_path)
{
    mode_t permission = 0666;
    if (mkfifo(tracepipe_path, permission) == -1) {
        if (errno != EEXIST) {
            spdlog::error("Failed to create tracepipe: {}", strerror(errno));
            return 1;
        }
    } else {
        spdlog::info("Trace pipe created");
    }

    int fd = open(tracepipe_path, O_RDONLY);
    if (fd == -1) {
        spdlog::error(
       "Failed to open tracepipe: {}",
            strerror(errno));
        return 2;
    }

    while (1) {
        char data[STR_MAX];
        ssize_t ret = read(fd, data, strnlen(data, STR_MAX));
        if (ret == -1) {
            spdlog::error(
                "Failed to read from tracepipe: {}",
                strerror(errno));
            return 3;
        } else {
            if (ret > 0) {
                std::cout << data;
            }
        }
    }
    return 0;
}


int main(int argc, const char **argv)
{
	spdlog::cfg::load_env_levels();
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

	argparse::ArgumentParser load_command("load");

	load_command.add_description(
		"Start an application with bpftime-server injected");
	load_command.add_argument("COMMAND")
		.help("Command to run")
		.nargs(argparse::nargs_pattern::at_least_one)
		.remaining();

	argparse::ArgumentParser start_command("start");

	start_command.add_description(
		"Start an application with bpftime-agent injected");
	start_command.add_argument("-s", "--enable-syscall-trace")
		.help("Whether to enable syscall trace")
		.flag();
	start_command.add_argument("COMMAND")
		.nargs(argparse::nargs_pattern::at_least_one)
		.remaining()
		.help("Command to run");
    start_command.add_argument("-p", "--print-to-trace-pipe")
        .help("Whether to send output of bpf_printk to the tracepipe")
        .flag();

	argparse::ArgumentParser attach_command("attach");

	attach_command.add_description("Inject bpftime-agent to a certain pid");
	attach_command.add_argument("-s", "--enable-syscall-trace")
		.help("Whether to enable syscall trace")
		.flag();
	attach_command.add_argument("PID").scan<'i', int>();

    argparse::ArgumentParser trace_command("trace");
	trace_command.add_description("Read contents of tracepipe");

	program.add_subparser(load_command);
	program.add_subparser(start_command);
	program.add_subparser(attach_command);
	program.add_subparser(trace_command);
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
		return run_command(executable_path.c_str(),
		                   extra_args,
				           so_path.c_str(),
				           nullptr,
				           nullptr);
	} else if (program.is_subcommand_used("start")) {
		auto agent_path = install_path / "libbpftime-agent.so";
		if (!std::filesystem::exists(agent_path)) {
			spdlog::error("Library not found: {}",
				      agent_path.c_str());
			return 1;
		}
		auto [executable_path, extra_args] =
			extract_path_and_args(start_command);
        std::string tracepipe_path;
        if (start_command.get<bool>("print-to-trace-pipe")) {
            tracepipe_path = install_path / "tracepipe";
            if (!std::filesystem::exists(tracepipe_path)) {
                spdlog::error("Tracepipe not found: {}",
                        tracepipe_path.c_str());
            }
        }
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
					   agent_path.c_str(),
					   tracepipe_path.c_str());
		} else {
			return run_command(executable_path.c_str(), extra_args,
					   agent_path.c_str(), nullptr,
					   tracepipe_path.c_str());
		}
	} else if (program.is_subcommand_used("attach")) {
		auto agent_path = install_path / "libbpftime-agent.so";
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
	} else if (program.is_subcommand_used("trace")) {
		auto tracepipe_path = install_path / "tracepipe";
		if (!std::filesystem::exists(tracepipe_path)) {
            spdlog::error("Tracepipe not found: {}",
                    tracepipe_path.c_str());
        }
        return read_tracepipe(tracepipe_path.c_str());
    }
	return 0;
}
