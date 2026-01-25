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
#include <atomic>
#include <optional>
#include <array>
#include <cctype>
#include <fstream>
#include <sstream>
#include <boost/interprocess/shared_memory_object.hpp>
#if __linux__
#include <grp.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/uio.h>
#include <sys/stat.h>
#endif
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
const char *strchrnul(const char *s, int c)
{
	while (*s && *s != (char)c) {
		s++;
	}
	return s;
}
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

static std::atomic<bool> g_trace_interrupted{ false };
static std::atomic<pid_t> g_trace_target_pid{ -1 };
static std::atomic<pid_t> g_trace_loader_pid{ -1 };

#if __linux__
static void trace_sigint_handler(int)
{
	g_trace_interrupted.store(true, std::memory_order_release);
	pid_t target = g_trace_target_pid.load(std::memory_order_acquire);
	if (target > 0) {
		kill(target, SIGUSR1);
	}
}
#endif

static bool str_starts_with(const char *main, const char *pat)
{
	if (strstr(main, pat) == main)
		return true;
	return false;
}

static int run_command(const char *path, const std::vector<std::string> &argv,
		       const char *ld_preload, const char *agent_so)
{
	int pid = fork();
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
						"Program exited abnormally, code={}",
						exit_code);
					return 1;
				}
			}
		}
	}
	return 1;
}

static int spawn_command(const char *path, const std::vector<std::string> &argv,
			 const char *ld_preload, const char *agent_so,
			 std::optional<std::pair<uid_t, gid_t>> drop_to)
{
	int pid = fork();
	if (pid == 0) {
#if __linux__
			if (drop_to && geteuid() == 0) {
				auto [uid, gid] = *drop_to;
				if (setgroups(0, nullptr) != 0) {
				}
				if (setgid(gid) != 0) {
				}
				if (setuid(uid) != 0) {
				}
			}
#else
			(void)drop_to;
#endif
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

		env_arr.push_back(nullptr);
		std::vector<const char *> argv_arr;
		argv_arr.push_back(path);
		for (const auto &str : argv)
			argv_arr.push_back(str.c_str());
		argv_arr.push_back(nullptr);
		execvpe(path, (char *const *)argv_arr.data(),
			(char *const *)env_arr.data());
		std::exit(127);
	}
	subprocess_pid = pid;
	return pid;
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

#if __linux__
static bool make_agent_ipc_addr(int pid, sockaddr_un &addr, socklen_t &len)
{
	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	std::string name = "bpftime-agent-" + std::to_string(pid);
	if (name.size() + 1 > sizeof(addr.sun_path))
		return false;
	addr.sun_path[0] = '\0';
	memcpy(addr.sun_path + 1, name.data(), name.size());
	len = (socklen_t)(offsetof(sockaddr_un, sun_path) + 1 + name.size());
	return true;
}

static bool try_send_agent_ipc(int pid, const std::string &request,
			       std::string *response_out = nullptr)
{
	int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
	if (fd < 0)
		return false;
	sockaddr_un addr {};
	socklen_t len = 0;
	if (!make_agent_ipc_addr(pid, addr, len)) {
		::close(fd);
		return false;
	}
	if (::connect(fd, (sockaddr *)&addr, len) != 0) {
		::close(fd);
		return false;
	}
	const char *buf = request.data();
	size_t left = request.size();
	while (left > 0) {
		ssize_t n = ::send(fd, buf, left, MSG_NOSIGNAL);
		if (n < 0) {
			::close(fd);
			return false;
		}
		buf += (size_t)n;
		left -= (size_t)n;
	}
	::shutdown(fd, SHUT_WR);
	if (response_out) {
		std::string resp;
		char rbuf[4096];
		for (;;) {
			ssize_t n = ::recv(fd, rbuf, sizeof(rbuf), 0);
			if (n <= 0)
				break;
			resp.append(rbuf, rbuf + n);
		}
		*response_out = std::move(resp);
	}
	::close(fd);
	return true;
}
#endif

static bool is_all_digits(const std::string &s)
{
	return !s.empty() &&
	       std::all_of(s.begin(), s.end(),
			   [](unsigned char c) { return std::isdigit(c); });
}

static std::vector<int> find_pids_by_comm(const std::string &comm_name)
{
	std::vector<int> matches;
#if __linux__
	std::error_code ec;
	for (const auto &entry :
	     std::filesystem::directory_iterator("/proc", ec)) {
		if (ec)
			break;
		if (!entry.is_directory())
			continue;
		auto leaf = entry.path().filename().string();
		if (!is_all_digits(leaf))
			continue;
		int pid = 0;
		try {
			pid = std::stoi(leaf);
		} catch (...) {
			continue;
		}
		std::ifstream ifs(entry.path() / "comm");
		if (!ifs.is_open())
			continue;
		std::string comm;
		if (!std::getline(ifs, comm))
			continue;
		if (comm == comm_name)
			matches.push_back(pid);
	}
#else
	(void)comm_name;
#endif
	return matches;
}

static std::optional<std::pair<uid_t, gid_t>> get_process_uid_gid(int pid)
{
#if __linux__
	std::ifstream ifs("/proc/" + std::to_string(pid) + "/status");
	if (!ifs.is_open())
		return std::nullopt;
	std::string line;
	uid_t uid = (uid_t)-1;
	gid_t gid = (gid_t)-1;
	while (std::getline(ifs, line)) {
		if (line.rfind("Uid:", 0) == 0) {
			std::istringstream iss(line.substr(4));
			unsigned long r = 0;
			if (iss >> r)
				uid = (uid_t)r;
		} else if (line.rfind("Gid:", 0) == 0) {
			std::istringstream iss(line.substr(4));
			unsigned long r = 0;
			if (iss >> r)
				gid = (gid_t)r;
		}
		if (uid != (uid_t)-1 && gid != (gid_t)-1)
			break;
	}
	if (uid == (uid_t)-1 || gid == (gid_t)-1)
		return std::nullopt;
	return std::make_pair(uid, gid);
#else
	(void)pid;
	return std::nullopt;
#endif
}

static const char *get_global_shm_name_for_cli()
{
	const char *name = getenv("BPFTIME_GLOBAL_SHM_NAME");
	if (name == nullptr || name[0] == '\0')
		return "bpftime_maps_shm";
	return name;
}

#if __linux__
static void try_relax_global_shm_permissions_for_target()
{
	std::string path = std::string("/dev/shm/") + get_global_shm_name_for_cli();
	::chmod(path.c_str(), 0666);
}
#endif

static std::optional<std::filesystem::path> current_exe_path()
{
#if __linux__
	std::array<char, 4096> buf {};
	ssize_t n = readlink("/proc/self/exe", buf.data(), buf.size() - 1);
	if (n <= 0)
		return std::nullopt;
	buf[(std::size_t)n] = '\0';
	return std::filesystem::path(buf.data());
#else
	return std::nullopt;
#endif
}

static std::optional<std::filesystem::path> find_build_root_from_exe()
{
	auto exe = current_exe_path();
	if (!exe)
		return std::nullopt;
	auto dir = exe->parent_path();
	for (int i = 0; i < 6; i++) {
		auto candidate = dir;
		for (int up = 0; up < i; up++)
			candidate = candidate.parent_path();
		if (candidate.empty())
			continue;
		auto agent = candidate / "runtime" / "agent" / AGENT_LIBRARY;
		auto server = candidate / "runtime" / "syscall-server" /
			      SYSCALL_SERVER_LIBRARY;
		if (std::filesystem::exists(agent) &&
		    std::filesystem::exists(server)) {
			return candidate;
		}
	}
	return std::nullopt;
}

static std::optional<std::filesystem::path> readlink_path(const std::string &p)
{
#if __linux__
	std::array<char, 4096> buf {};
	ssize_t n = readlink(p.c_str(), buf.data(), buf.size() - 1);
	if (n <= 0)
		return std::nullopt;
	buf[(std::size_t)n] = '\0';
	return std::filesystem::path(buf.data());
#else
	(void)p;
	return std::nullopt;
#endif
}

static std::string resolve_cuobjdump_path()
{
	const auto exists = [](const std::filesystem::path &p) -> bool {
		std::error_code ec;
		return std::filesystem::exists(p, ec);
	};

	if (const char *p = getenv("BPFTIME_CUOBJDUMP"); p && p[0] != '\0') {
		std::string s(p);
		if (s.find('/') == std::string::npos)
			return s;
		if (exists(s))
			return s;
	}

	const auto try_root = [&](const char *env) -> std::optional<std::string> {
		if (const char *p = getenv(env); p && p[0] != '\0') {
			auto cand = std::filesystem::path(p) / "bin" / "cuobjdump";
			if (exists(cand))
				return cand.string();
		}
		return std::nullopt;
	};

	for (const char *env : {"BPFTIME_CUDA_ROOT", "CUDA_HOME", "CUDA_PATH",
				"LLVMBPF_CUDA_PATH", "CUDAToolkit_ROOT"}) {
		if (auto p = try_root(env))
			return *p;
	}

#if __linux__
	{
		std::error_code ec;
		const std::filesystem::path usr_local("/usr/local");
		for (const auto &entry :
		     std::filesystem::directory_iterator(usr_local, ec)) {
			if (ec)
				break;
			if (!entry.is_directory())
				continue;
			auto name = entry.path().filename().string();
			if (name != "cuda" && !name.starts_with("cuda-"))
				continue;
			auto cand = entry.path() / "bin" / "cuobjdump";
			if (exists(cand))
				return cand.string();
		}
	}
#endif

	return "cuobjdump";
}

static std::optional<std::filesystem::path> prepare_cuda_late_ptx_dir(
	int pid, std::optional<std::pair<uid_t, gid_t>> target_uid_gid)
{
#if __linux__
	auto exe = readlink_path("/proc/" + std::to_string(pid) + "/exe");
	if (!exe)
		return std::nullopt;

	auto read_cmdline_candidates = [&](const std::filesystem::path &primary)
		-> std::vector<std::filesystem::path> {
		std::vector<std::filesystem::path> out;
		out.push_back(primary);
		std::ifstream ifs("/proc/" + std::to_string(pid) + "/cmdline",
				  std::ios::binary);
		if (!ifs.is_open())
			return out;
		std::ostringstream oss;
		oss << ifs.rdbuf();
		std::string buf = oss.str();
		size_t pos = 0;
		while (pos < buf.size()) {
			size_t end = buf.find('\0', pos);
			if (end == std::string::npos)
				end = buf.size();
			std::string tok = buf.substr(pos, end - pos);
			pos = end + 1;
			if (tok.empty())
				continue;
			if (tok[0] == '-')
				continue;
			std::filesystem::path p(tok);
			std::error_code ec;
			if (!std::filesystem::exists(p, ec))
				continue;
			if (!std::filesystem::is_regular_file(p, ec))
				continue;
			if (p == primary)
				continue;
			out.push_back(p);
		}
		return out;
	};

	char tmp_template[] = "/tmp/bpftime-late-ptx.XXXXXX";
	char *dir_c = mkdtemp(tmp_template);
	if (dir_c == nullptr)
		return std::nullopt;

	std::filesystem::path dir(dir_c);
	chmod(dir.c_str(), 0755);
	if (geteuid() == 0 && target_uid_gid) {
		if (chown(dir.c_str(), target_uid_gid->first, target_uid_gid->second) !=
		    0) {
		}
	}

	const auto sh_quote = [](const std::string &s) -> std::string {
		std::string out;
		out.reserve(s.size() + 8);
		out.push_back('\'');
		for (char c : s) {
			if (c == '\'')
				out += "'\"'\"'";
			else
				out.push_back(c);
		}
		out.push_back('\'');
		return out;
	};

	const auto cuobjdump = resolve_cuobjdump_path();
	auto run_extract = [&](const std::filesystem::path &target_exe) -> bool {
		std::string cmd;
		cmd.reserve(512);
		cmd += "cd -- ";
		cmd += sh_quote(dir.string());
		cmd += " && ";
		cmd += "LD_PRELOAD= ";
		cmd += sh_quote(cuobjdump);
		cmd += " --extract-ptx all ";
		cmd += sh_quote(target_exe.string());

		std::vector<std::string> arg_strs;
		arg_strs.emplace_back("sh");
		arg_strs.emplace_back("-c");
		arg_strs.emplace_back(std::move(cmd));
		std::vector<char *> argv;
		argv.reserve(arg_strs.size() + 1);
		for (auto &s : arg_strs)
			argv.push_back(s.data());
		argv.push_back(nullptr);

		std::vector<std::string> env_strs;
		for (char **p = ::environ; p && *p; ++p) {
			if (strncmp(*p, "LD_PRELOAD=", 11) == 0)
				continue;
			env_strs.emplace_back(*p);
		}
		env_strs.emplace_back("LD_PRELOAD=");
		std::vector<char *> envp;
		envp.reserve(env_strs.size() + 1);
		for (auto &s : env_strs)
			envp.push_back(s.data());
		envp.push_back(nullptr);

		pid_t child_pid = -1;
		int rc = posix_spawnp(&child_pid, "/bin/sh", nullptr, nullptr,
				      argv.data(), envp.data());
		if (rc != 0)
			return false;
		int status = 0;
		if (waitpid(child_pid, &status, 0) < 0)
			return false;
		if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
			return false;
		return true;
	};

	for (const auto &cand : read_cmdline_candidates(*exe)) {
		std::error_code ec;
		for (const auto &entry :
		     std::filesystem::directory_iterator(dir, ec)) {
			if (ec)
				break;
			if (entry.is_regular_file() &&
			    entry.path().string().ends_with(".ptx")) {
				std::filesystem::remove(entry.path(), ec);
			}
		}
		if (!run_extract(cand))
			continue;
		size_t ptx_count = 0;
		for (const auto &entry :
		     std::filesystem::directory_iterator(dir, ec)) {
			if (ec)
				break;
			if (!entry.is_regular_file())
				continue;
			if (entry.path().string().ends_with(".ptx"))
				ptx_count++;
		}
		if (ptx_count > 0) {
			spdlog::info(
				"trace: extracted {} PTX file(s) to {}",
				ptx_count, dir.string());
			return dir;
		}
	}
	return std::nullopt;
#else
	(void)pid;
	(void)target_uid_gid;
	return std::nullopt;
#endif
}

static std::filesystem::path resolve_library_path_or_exit(
	const std::filesystem::path &install_path,
	const std::filesystem::path &install_relative,
	const std::filesystem::path &build_relative, const char *what)
{
	if (auto build_root = find_build_root_from_exe(); build_root) {
		auto b = *build_root / build_relative;
		if (std::filesystem::exists(b)) {
			spdlog::info("Using {} from build tree: {}", what,
				     b.c_str());
			return b;
		}
	}
	auto p = install_path / install_relative;
	if (std::filesystem::exists(p))
		return p;
	spdlog::error("Library not found for {}: {}", what, p.c_str());
	std::exit(1);
}

static std::pair<std::string, std::vector<std::string>>
extract_path_and_args(const argparse::ArgumentParser &parser)
{
	std::vector<std::string> items;
	try {
		items = parser.get<std::vector<std::string>>("COMMAND");
	} catch (std::logic_error &err) {
		std::cerr << parser;
		exit(1);
	}
	std::string executable = items[0];
	items.erase(items.begin());
	return { executable, items };
}

static void signal_handler(int sig)
{
	if (subprocess_pid) {
		kill(subprocess_pid, sig);
	}
}

int main(int argc, const char **argv)
{
	const auto agent_config = bpftime::construct_agent_config_from_env();
	(void)agent_config;
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

	argparse::ArgumentParser attach_command("attach");

	attach_command.add_description("Inject bpftime-agent to a certain pid");
	attach_command.add_argument("-s", "--enable-syscall-trace")
		.help("Whether to enable syscall trace")
		.flag();
	attach_command.add_argument("PID").scan<'i', int>();

	argparse::ArgumentParser detach_command("detach");
	detach_command.add_description("Detach all attached agents");

	argparse::ArgumentParser trace_command("trace");
	trace_command.add_description(
		"Run a loader with bpftime-server injected, then attach bpftime-agent to a running process");
	trace_command.add_argument("--pid")
		.help("Target PID to attach bpftime-agent")
		.default_value(-1)
		.scan<'i', int>();
	trace_command.add_argument("--pidof")
		.help("Resolve PID by /proc/<pid>/comm name (Linux only); must match exactly and be unique")
		.default_value(std::string());
	trace_command.add_argument("--auto-refresh-ms")
		.help("Agent-side auto refresh interval (ms) for discovering late-loaded links")
		.default_value(500)
		.scan<'i', int>();
	trace_command.add_argument("COMMAND")
		.nargs(argparse::nargs_pattern::at_least_one)
		.remaining()
		.help("Loader command to run (will be injected with bpftime-server)");

	program.add_subparser(load_command);
	program.add_subparser(start_command);
	program.add_subparser(attach_command);
	program.add_subparser(detach_command);
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
		auto so_path = resolve_library_path_or_exit(
			install_path, SYSCALL_SERVER_LIBRARY,
			std::filesystem::path("runtime") / "syscall-server" /
				SYSCALL_SERVER_LIBRARY,
			"syscall-server");
		auto [executable_path, extra_args] =
			extract_path_and_args(load_command);
		return run_command(executable_path.c_str(), extra_args,
				   so_path.c_str(), nullptr);
	} else if (program.is_subcommand_used("start")) {
		auto agent_path = resolve_library_path_or_exit(
			install_path, AGENT_LIBRARY,
			std::filesystem::path("runtime") / "agent" /
				AGENT_LIBRARY,
			"agent");
		auto [executable_path, extra_args] =
			extract_path_and_args(start_command);
		if (start_command.get<bool>("enable-syscall-trace")) {
			auto transformer_path = resolve_library_path_or_exit(
				install_path, AGENT_TRANSFORMER_LIBRARY,
				std::filesystem::path("attach") /
					"text_segment_transformer" /
					AGENT_TRANSFORMER_LIBRARY,
				"agent-transformer");
			return run_command(executable_path.c_str(), extra_args,
					   transformer_path.c_str(),
					   agent_path.c_str());
		} else {
			return run_command(executable_path.c_str(), extra_args,
					   agent_path.c_str(), nullptr);
		}
	} else if (program.is_subcommand_used("attach")) {
		auto agent_path = resolve_library_path_or_exit(
			install_path, AGENT_LIBRARY,
			std::filesystem::path("runtime") / "agent" /
				AGENT_LIBRARY,
			"agent");
		auto pid = attach_command.get<int>("PID");
		if (attach_command.get<bool>("enable-syscall-trace")) {
			auto transformer_path = resolve_library_path_or_exit(
				install_path, AGENT_TRANSFORMER_LIBRARY,
				std::filesystem::path("attach") /
					"text_segment_transformer" /
					AGENT_TRANSFORMER_LIBRARY,
				"agent-transformer");
			return inject_by_frida(pid, transformer_path.c_str(),
					       agent_path.c_str());
		} else {
			return inject_by_frida(pid, agent_path.c_str(), "");
		}
	} else if (program.is_subcommand_used("trace")) {
		auto pid = trace_command.get<int>("pid");
		auto pidof_name = trace_command.get<std::string>("pidof");
		if (pid > 0 && !pidof_name.empty()) {
			spdlog::error(
				"trace: --pid and --pidof are mutually exclusive");
			return 1;
		}
		if (!pidof_name.empty()) {
			auto matches = find_pids_by_comm(pidof_name);
			if (matches.empty()) {
				spdlog::error(
					"trace: no process matched --pidof {}",
					pidof_name);
				return 1;
			}
			if (matches.size() != 1) {
				spdlog::error(
					"trace: --pidof {} matched {} processes, please specify --pid instead",
					pidof_name, matches.size());
				for (auto p : matches) {
					spdlog::error("matched pid={}", p);
				}
				return 1;
			}
			pid = matches[0];
			spdlog::info("Resolved --pidof {} -> pid {}",
				     pidof_name, pid);
		}
		if (pid <= 0) {
			spdlog::error(
				"trace: you must specify either --pid <PID> or --pidof <COMM>");
			return 1;
		}
		std::optional<std::pair<uid_t, gid_t>> drop_to;
#if __linux__
		if (geteuid() == 0) {
			if (auto ug = get_process_uid_gid(pid); ug) {
				drop_to = ug;
				spdlog::info(
					"trace: will run loader as target uid={}, gid={}",
					(unsigned long)ug->first,
					(unsigned long)ug->second);
				try_relax_global_shm_permissions_for_target();
			} else {
				spdlog::warn(
					"trace: unable to resolve target uid/gid; loader will run as root (may break shm permissions)");
			}
		}
#endif

		auto refresh_ms = trace_command.get<int>("auto-refresh-ms");
		auto late_ptx_dir = prepare_cuda_late_ptx_dir(pid, drop_to);
		auto agent_path = resolve_library_path_or_exit(
			install_path, AGENT_LIBRARY,
			std::filesystem::path("runtime") / "agent" /
				AGENT_LIBRARY,
			"agent");
		auto server_path = resolve_library_path_or_exit(
			install_path, SYSCALL_SERVER_LIBRARY,
			std::filesystem::path("runtime") / "syscall-server" /
				SYSCALL_SERVER_LIBRARY,
			"syscall-server");
		spdlog::info("trace: using syscall-server `{}`",
			     server_path.string());
		spdlog::info("trace: using agent `{}`", agent_path.string());

		auto [executable_path, extra_args] =
			extract_path_and_args(trace_command);
		int child_pid = spawn_command(executable_path.c_str(), extra_args,
					      server_path.c_str(), nullptr,
					      drop_to);
		if (child_pid <= 0) {
			spdlog::error("Failed to spawn loader process");
			return 1;
		}
		spdlog::info("trace: loader pid={}", child_pid);

#if __linux__
		g_trace_target_pid.store(pid, std::memory_order_release);
		g_trace_loader_pid.store(child_pid, std::memory_order_release);
		g_trace_interrupted.store(false, std::memory_order_release);
		struct sigaction sa {};
		sa.sa_handler = trace_sigint_handler;
		sigemptyset(&sa.sa_mask);
		sa.sa_flags = 0;
		struct sigaction old_int {};
		struct sigaction old_term {};
		sigaction(SIGINT, &sa, &old_int);
		sigaction(SIGTERM, &sa, &old_term);
#endif

		std::string agent_arg = "loader_pid=" + std::to_string(child_pid);
		agent_arg += ";force_reinit=1";
		if (refresh_ms > 0) {
			agent_arg +=
				";auto_refresh_ms=" + std::to_string(refresh_ms);
		}
		if (late_ptx_dir) {
			agent_arg += ";cuda_late_ptx_dir=" + late_ptx_dir->string();
		} else {
			agent_arg += ";cuda_disable_cuobjdump=1";
		}

		int rc = 0;
		bool agent_attached = false;
#if __linux__
		std::string req = "refresh " + agent_arg;
		if (try_send_agent_ipc(pid, req)) {
			spdlog::info("trace: refreshed existing agent via IPC");
			agent_attached = true;
		} else
#endif
		{
			rc = inject_by_frida(pid, agent_path.c_str(),
					     agent_arg.c_str());
			agent_attached = (rc == 0);
		}

		int status = 0;
#if __linux__
		bool stop_requested = false;
		for (;;) {
			pid_t w = waitpid(child_pid, &status, 0);
			if (w == child_pid)
				break;
			if (w < 0 && errno == EINTR) {
				if (g_trace_interrupted.load(
					    std::memory_order_acquire)) {
					if (!stop_requested) {
						stop_requested = true;
						if (agent_attached) {
							try_send_agent_ipc(pid,
									   "detach");
							kill(pid, SIGUSR1);
							usleep(200 * 1000);
						}
						kill(child_pid, SIGINT);
					}
					continue;
				}
				continue;
			}
			break;
		}

		if (agent_attached)
			try_send_agent_ipc(pid, "detach");
		if (agent_attached) {
			kill(pid, SIGUSR1);
			usleep(200 * 1000);
		}
		sigaction(SIGINT, &old_int, nullptr);
		sigaction(SIGTERM, &old_term, nullptr);
#else
		waitpid(child_pid, &status, 0);
#endif
		return rc;
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
		bool sent = false;
		bpftime::shm_holder.global_shared_memory
			.iterate_all_pids_in_alive_agent_set([&](int pid) {
#if __linux__
				try_send_agent_ipc(pid, "detach");
#endif
				SPDLOG_INFO("Delivering SIGUSR1 to {}", pid);
				int err = kill(pid, SIGUSR1);
				if (err < 0) {
					SPDLOG_WARN(
						"Unable to signal process {}: {}",
						pid, strerror(errno));
				}
				sent = true;
			});
		if (!sent) {
			SPDLOG_INFO("No process was signaled.");
		}
	}
	return 0;
}
