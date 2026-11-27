/* SPDX-License-Identifier: MIT
 *
 * Linux target backend: real /proc inspection, on-disk ELF fetch, and
 * LD_PRELOAD spawn with SIGSTOP hold / SIGCONT resume.
 */
#ifdef __linux__
#include "platform_target.hpp"

#include "bpftime_agent_ipc.hpp"

#include <fcntl.h>
#include <signal.h>
#include <spdlog/spdlog.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace bpftime
{
namespace remote
{

namespace
{
// Reap exited children without blocking so short workloads do not turn into
// zombies. WNOHANG never consumes a stop notification, so the WUNTRACED wait in
// platform_spawn still observes the held child.
void reap_children(int)
{
	int saved = errno;
	while (waitpid(-1, nullptr, WNOHANG) > 0) {
	}
	errno = saved;
}

std::string read_small_file(const std::string &path)
{
	std::ifstream f(path);
	if (!f)
		return {};
	std::stringstream ss;
	ss << f.rdbuf();
	std::string s = ss.str();
	while (!s.empty() && (s.back() == '\n' || s.back() == '\r'))
		s.pop_back();
	return s;
}
} // namespace

PlatformIdentity platform_identity()
{
#if defined(__aarch64__) || defined(_M_ARM64)
	const char *arch = "arm64";
#elif defined(__x86_64__)
	const char *arch = "x86_64";
#else
	const char *arch = "unknown";
#endif
	return { "linux", arch, (int)sysconf(_SC_NPROCESSORS_ONLN) };
}

void platform_init()
{
	struct sigaction sa {};
	sa.sa_handler = reap_children;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_RESTART | SA_NOCLDSTOP;
	sigaction(SIGCHLD, &sa, nullptr);
}

std::string platform_default_agent_path(const char *argv0)
{
	if (const char *env = getenv("BPFTIME_AGENT_LIB"); env && env[0])
		return env;
	char resolved[4096] = {};
	if (realpath(argv0, resolved) == nullptr)
		return "libbpftime-agent.so";
	std::string exe(resolved);
	size_t slash = exe.find_last_of('/');
	std::string dir = slash == std::string::npos ? "." :
						       exe.substr(0, slash);
	// Build-tree: <build>/tools/bpftime_remote_host/bpftime-remote-host
	return dir + "/../../runtime/agent/libbpftime-agent.so";
}

const char *platform_preload_env_name()
{
	return "LD_PRELOAD";
}

ProcCheckResult platform_proc_check(pid_t pid)
{
	ProcCheckResult r;
	if (pid <= 0 || kill(pid, 0) != 0)
		return r;
	r.alive = true;
	char buf[4096] = {};
	ssize_t n = readlink(
		("/proc/" + std::to_string(pid) + "/exe").c_str(), buf,
		sizeof(buf) - 1);
	if (n > 0)
		r.exe.assign(buf, (size_t)n);
	r.comm = read_small_file("/proc/" + std::to_string(pid) + "/comm");
	if (r.comm.empty() && !r.exe.empty()) {
		size_t slash = r.exe.find_last_of('/');
		r.comm = slash == std::string::npos ? r.exe :
						      r.exe.substr(slash + 1);
	}
	r.status = read_small_file("/proc/" + std::to_string(pid) + "/status");
	return r;
}

std::string platform_proc_maps(pid_t pid)
{
	return read_small_file("/proc/" + std::to_string(pid) + "/maps");
}

std::vector<uint8_t> platform_fetch_elf(const std::string &path,
					std::string &canonical_path)
{
	std::vector<uint8_t> bytes;
	int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
	if (fd < 0)
		return bytes;
	struct stat st {};
	if (fstat(fd, &st) != 0 || st.st_size < 4) {
		close(fd);
		return bytes;
	}
	bytes.resize((size_t)st.st_size);
	size_t off = 0;
	while (off < bytes.size()) {
		ssize_t n = read(fd, bytes.data() + off, bytes.size() - off);
		if (n == 0)
			break;
		if (n < 0) {
			if (errno == EINTR)
				continue;
			close(fd);
			bytes.clear();
			return bytes;
		}
		off += (size_t)n;
	}
	close(fd);
	bytes.resize(off);
	static constexpr unsigned char kElfMagic[4] = { 0x7f, 'E', 'L', 'F' };
	if (bytes.size() < 4 || memcmp(bytes.data(), kElfMagic, 4) != 0) {
		SPDLOG_WARN("fetch_elf: {} is not an ELF file", path);
		bytes.clear();
		return bytes;
	}
	canonical_path = path;
	return bytes;
}

bool platform_translate_uprobe(std::string &, uint64_t &)
{
	return false;
}

pid_t platform_spawn(char *const argv[], char *const envp[],
		     const std::string &cwd, bool hold)
{
	pid_t pid = fork();
	if (pid < 0)
		throw std::runtime_error(std::string("fork failed: ") +
					 strerror(errno));
	if (pid == 0) {
		if (!cwd.empty())
			(void)chdir(cwd.c_str());
		if (hold)
			raise(SIGSTOP);
		execve(argv[0], argv, envp);
		_exit(127);
	}
	if (hold) {
		int status = 0;
		pid_t r;
		do {
			r = waitpid(pid, &status, WUNTRACED);
		} while (r < 0 && errno == EINTR);
		if (r != pid || !WIFSTOPPED(status))
			throw std::runtime_error(
				"child did not stop before entry");
		SPDLOG_DEBUG("spawn: pid {} stopped awaiting resume", pid);
	}
	return pid;
}

void platform_resume(pid_t pid)
{
	if (pid > 0)
		kill(pid, SIGCONT);
}

void platform_refresh_agent(pid_t pid)
{
	if (pid > 0)
		(void)bpftime::try_send_agent_ipc(pid, "refresh");
}

} // namespace remote
} // namespace bpftime

#endif // __linux__
