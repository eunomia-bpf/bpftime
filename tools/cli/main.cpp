#include "bpftime_shm.hpp"
#include <bpftime_shm_internal.hpp>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <exception>
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
#include <spdlog/spdlog.h>
#include <spdlog/cfg/env.h>
#ifdef __APPLE__
#include <crt_externs.h>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <unistd.h>
inline char **get_environ()
{
	return *_NSGetEnviron();
}
constexpr const char *AGENT_LIBRARY = "libbpftime-agent.dylib";
constexpr const char *SYSCALL_SERVER_LIBRARY =
	"libbpftime-syscall-server.dylib";
constexpr const char *AGENT_TRANSFORMER_LIBRARY =
	"libbpftime-agent-transformer.dylib";
#if __APPLE__
// macOS has strchrnul in system headers, but it returns char* instead of const
// char* We need a const version for our use case
static inline const char *strchrnul_const(const char *s, int c)
{
	while (*s && *s != (char)c) {
		s++;
	}
	return s;
}
#define strchrnul strchrnul_const
#else
const char *strchrnul(const char *s, int c)
{
	while (*s && *s != (char)c) {
		s++;
	}
	return s;
}
#endif
int execvpe(const char *file, char *const argv[], char *const envp[])
{
	for (const char *path = getenv("PATH"); path && *path;
	     path = strchr(path, ':') + 1) {
		char buf[PATH_MAX];
		const char *end = strchrnul(path, ':');
		size_t len = end - path;
		memcpy(buf, path, len);
		buf[len] = '/';
		strcpy(buf + len + 1, file);
		execve(buf, argv, envp);
		if (errno != ENOENT)
			return -1;
	}
	errno = ENOENT;
	return -1;
}
#elif __linux__
extern char **environ;
constexpr const char *AGENT_LIBRARY = "libbpftime-agent.so";
constexpr const char *SYSCALL_SERVER_LIBRARY = "libbpftime-syscall-server.so";
constexpr const char *AGENT_TRANSFORMER_LIBRARY =
	"libbpftime-agent-transformer.so";
#else
#error "Unsupported Platform"
#endif

static int subprocess_pid = 0;

static bool str_starts_with(const char *main, const char *pat)
{
	if (strstr(main, pat) == main)
		return true;
	return false;
}

static int run_command(const char *path, const std::vector<std::string> &argv,
		       const char *ld_preload, const char *agent_so,
		       const std::vector<std::string> &env_args)
{
	int pid = fork();
	if (pid < 0) {
		spdlog::error("fork failed: {}", strerror(errno));
		return 1;
	}
	if (pid == 0) {
		std::string ld_preload_str("LD_PRELOAD=");
		std::string agent_so_str("AGENT_SO=");
		ld_preload_str += ld_preload;

		if (agent_so) {
			agent_so_str += agent_so;
		}
		std::vector<const char *> env_arr;
#if __APPLE__
		char **p = get_environ();
#else
		char **p = environ;
#endif
		while (*p) {
			env_arr.push_back(*p);
			p++;
		}
		bool ld_preload_set = false, agent_so_set = false;
		for (auto &s : env_arr) {
			if (str_starts_with(s, "LD_PRELOAD=")) {
				s = ld_preload_str.c_str();
				ld_preload_set = true;
			} else if (str_starts_with(s, "AGENT_SO=")) {
				s = agent_so_str.c_str();
				agent_so_set = true;
			}
		}
		if (!ld_preload_set)
			env_arr.push_back(ld_preload_str.c_str());
		if (!agent_so_set)
			env_arr.push_back(agent_so_str.c_str());
		for (const auto &env_arg : env_args) {
			auto key_end = env_arg.find('=');
			auto prefix = env_arg.substr(0, key_end + 1);
			bool replaced = false;
			for (auto &entry : env_arr) {
				if (str_starts_with(entry, prefix.c_str())) {
					entry = env_arg.c_str();
					replaced = true;
					break;
				}
			}
			if (!replaced) {
				env_arr.push_back(env_arg.c_str());
			}
		}
		env_arr.push_back(nullptr);
		std::vector<const char *> argv_arr;
		argv_arr.push_back(path);
		for (const auto &str : argv)
			argv_arr.push_back(str.c_str());
		argv_arr.push_back(nullptr);
		execvpe(path, (char *const *)argv_arr.data(),
			(char *const *)env_arr.data());
		spdlog::error("execvpe failed for {}: {}", path,
			      strerror(errno));
		_exit(errno == ENOENT ? 127 : 126);
	} else {
		subprocess_pid = pid;
		int status;
		if (int cid = waitpid(pid, &status, 0); cid > 0) {
			if (WIFEXITED(status)) {
				int exit_code = WEXITSTATUS(status);
				if (exit_code != 0) {
					spdlog::error(
						"Program exited abnormally, code={}",
						exit_code);
					return exit_code;
				}
				return 0;
			}
			if (WIFSIGNALED(status)) {
				int signal_code = WTERMSIG(status);
				spdlog::error("Program exited by signal {}",
					      signal_code);
				return 128 + signal_code;
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

static std::tuple<std::string, std::vector<std::string>,
		  std::vector<std::string>>
build_command_launch_args(const argparse::ArgumentParser &parser,
			  bool include_runtime_env_options,
			  bool include_kernel_loader_options)
{
	std::vector<std::string> items;
	std::vector<std::string> env_args;
	try {
		items = parser.get<std::vector<std::string>>("COMMAND");
		if (include_runtime_env_options) {
			if (parser.get<bool>("--no-jit")) {
				env_args.emplace_back(
					"BPFTIME_DISABLE_JIT=true");
			}
			if (include_kernel_loader_options &&
			    parser.get<bool>("--run-with-kernel-verifier")) {
				env_args.emplace_back(
					"BPFTIME_RUN_WITH_KERNEL=true");
			}
			if (include_kernel_loader_options &&
			    parser.is_used("--bpftime-not-load-pattern")) {
				env_args.emplace_back(
					"BPFTIME_NOT_LOAD_PATTERN=" +
					parser.get<std::string>(
						"--bpftime-not-load-pattern"));
			}
			if (parser.is_used("--spdlog-level")) {
				env_args.emplace_back(
					"SPDLOG_LEVEL=" +
					parser.get<std::string>(
						"--spdlog-level"));
			}
			if (parser.is_used("--bpftime-log-output")) {
				env_args.emplace_back(
					"BPFTIME_LOG_OUTPUT=" +
					parser.get<std::string>(
						"--bpftime-log-output"));
			}
			if (parser.get<bool>("--allow-external-maps")) {
				env_args.emplace_back(
					"BPFTIME_ALLOW_EXTERNAL_MAPS=true");
			}
			if (parser.is_used("--memory-size")) {
				env_args.emplace_back(
					"BPFTIME_SHM_MEMORY_MB=" +
					std::to_string(parser.get<int>(
						"--memory-size")));
			}
		}
	} catch (std::logic_error &err) {
		std::cerr << parser;
		exit(1);
	}
	std::string executable = items[0];
	items.erase(items.begin());
	return { executable, items, env_args };
}

static void
add_common_runtime_env_cli_options(argparse::ArgumentParser &command)
{
	command.add_argument("--no-jit")
		.help("Same as BPFTIME_DISABLE_JIT, disable JIT and use interpreter")
		.flag();
	command.add_argument("--spdlog-level")
		.help("Same as SPDLOG_LEVEL, control the log level dynamically. Available levels: trace, debug, info, warn, err, critical, off.");
	command.add_argument("--bpftime-log-output")
		.help("Same as BPFTIME_LOG_OUTPUT, control the log output destination, for example 'console' or a file path.");
	command.add_argument("--allow-external-maps")
		.help("Same as BPFTIME_ALLOW_EXTERNAL_MAPS, allow loading unsupported external maps with the bpftime syscall-server library.")
		.flag();
	command.add_argument("--memory-size")
		.help("Same as BPFTIME_SHM_MEMORY_MB, set the shared memory size for bpftime maps in MB.")
		.nargs(1)
		.scan<'i', int>();
}

static void add_kernel_loader_cli_options(argparse::ArgumentParser &command)
{
	command.add_argument("--run-with-kernel-verifier")
		.help("Same as BPFTIME_RUN_WITH_KERNEL, load the eBPF application with the kernel eBPF loader and kernel verifier.")
		.flag();
	command.add_argument("--bpftime-not-load-pattern")
		.help("Same as BPFTIME_NOT_LOAD_PATTERN, a regular expression used with BPFTIME_RUN_WITH_KERNEL to skip loading unsupported programs into the kernel.");
}

static void signal_handler(int sig)
{
	if (subprocess_pid) {
		kill(subprocess_pid, sig);
	}
}

int main(int argc, const char **argv)
{
	spdlog::cfg::load_env_levels();
	signal(SIGINT, signal_handler);
	signal(SIGTSTP, signal_handler);
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
		SPDLOG_WARN(
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

	load_command
		.add_description(
			"Start an application with bpftime-server injected")
		.add_epilog(
			"For more information and options, please see https://eunomia.dev/bpftime");
	load_command.add_argument("COMMAND")
		.help("Command to run")
		.nargs(argparse::nargs_pattern::at_least_one)
		.remaining();
	add_common_runtime_env_cli_options(load_command);
	add_kernel_loader_cli_options(load_command);

	argparse::ArgumentParser start_command("start");

	start_command
		.add_description(
			"Start an application with bpftime-agent injected")
		.add_epilog(
			"For more information and options, please see https://eunomia.dev/bpftime");
	start_command.add_argument("-s", "--enable-syscall-trace")
		.help("Whether to enable syscall trace")
		.flag();
	start_command.add_argument("COMMAND")
		.nargs(argparse::nargs_pattern::at_least_one)
		.remaining()
		.help("Command to run");

	argparse::ArgumentParser attach_command("attach");

	attach_command.add_description("Inject bpftime-agent to a certain pid")
		.add_epilog(
			"For more information and options, please see https://eunomia.dev/bpftime");
	attach_command.add_argument("-s", "--enable-syscall-trace")
		.help("Whether to enable syscall trace")
		.flag();
	attach_command.add_argument("PID").scan<'i', int>();

	argparse::ArgumentParser detach_command("detach");
	detach_command.add_description("Detach all attached agents")
		.add_epilog(
			"For more information and options, please see https://eunomia.dev/bpftime");

	program.add_subparser(load_command);
	program.add_subparser(start_command);
	program.add_subparser(attach_command);
	program.add_subparser(detach_command);
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
		auto so_path = install_path / SYSCALL_SERVER_LIBRARY;
		if (!std::filesystem::exists(so_path)) {
			spdlog::error("Library not found: {}", so_path.c_str());
			return 1;
		}
		auto [executable_path, extra_args, env_args] =
			build_command_launch_args(load_command, true, true);
		return run_command(executable_path.c_str(), extra_args,
				   so_path.c_str(), nullptr, env_args);
	} else if (program.is_subcommand_used("start")) {
		auto agent_path = install_path / AGENT_LIBRARY;
		if (!std::filesystem::exists(agent_path)) {
			spdlog::error("Library not found: {}",
				      agent_path.c_str());
			return 1;
		}
		auto [executable_path, extra_args, env_args] =
			build_command_launch_args(start_command, false, false);
		if (start_command.get<bool>("enable-syscall-trace")) {
			auto transformer_path =
				install_path /
				"libbpftime-agent-transformer.so";
			if (!std::filesystem::exists(transformer_path)) {
				spdlog::error("Library not found: {}",
					      transformer_path.c_str());
				return 1;
			}
			// transformer_path += ":/usr/lib/libclient.so";
			return run_command(executable_path.c_str(), extra_args,
					   transformer_path.c_str(),
					   agent_path.c_str(), env_args);
		} else {
			// agent_path += ":/usr/lib/libclient.so";
			return run_command(executable_path.c_str(), extra_args,
					   agent_path.c_str(), nullptr,
					   env_args);
		}
	} else if (program.is_subcommand_used("attach")) {
		auto agent_path = install_path / AGENT_LIBRARY;
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
	} else if (program.is_subcommand_used("detach")) {
		SPDLOG_DEBUG("Detaching..");
		try {
			bpftime_initialize_global_shm(
				bpftime::shm_open_type::SHM_OPEN_ONLY);
		} catch (std::exception &ex) {
			SPDLOG_WARN(
				"Shared memory not created, seems syscall server is not running");
			return 0;
		}
		bool sended = false;
		bpftime::shm_holder.global_shared_memory
			.iterate_all_pids_in_alive_agent_set([&](int pid) {
				SPDLOG_INFO("Delivering SIGUSR1 to {}", pid);
				int err = kill(pid, SIGUSR1);
				if (err < 0) {
					SPDLOG_WARN(
						"Unable to signal process {}: {}",
						pid, strerror(errno));
				}
				sended = true;
			});
		if (!sended) {
			SPDLOG_INFO("No process was signaled.");
		}
	}
	return 0;
}
