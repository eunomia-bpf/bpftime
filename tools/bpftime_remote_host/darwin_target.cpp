/* SPDX-License-Identifier: MIT
 *
 * Darwin target backend: synthesized proc maps, Mach-O -> stub ELF fetch, and
 * POSIX_SPAWN_START_SUSPENDED workload spawn with SIGCONT resume.
 */
#ifdef __APPLE__
#include "platform_target.hpp"
#include "elf_synth.hpp"
#include "macho_reader.hpp"

#include <frida-gum.h>
#include <libproc.h>
#include <signal.h>
#include <spawn.h>
#include <spdlog/spdlog.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>


namespace bpftime
{
namespace remote
{

namespace
{
std::mutex g_symbol_mutex;
std::unordered_map<std::string, ModuleSymbols> g_symbol_cache;
UmbrellaIndex g_umbrella;
bool g_umbrella_built = false;

const char *kLibSystemVirtual =
	"/bpftime-dyld/usr/lib/libSystem.B.dylib";
const char *kLibSystemReal = "/usr/lib/libSystem.B.dylib";

// Reap exited workloads so a finished child does not linger as a zombie (a
// zombie still answers kill(pid,0), which would break liveness reporting).
void reap_children(int)
{
	int saved = errno;
	while (waitpid(-1, nullptr, WNOHANG) > 0) {
	}
	errno = saved;
}

// Prefix dyld_shared_cache image paths with their /bpftime-dyld virtual form
// inside a synthesized /proc/<pid>/maps document.
std::string virtualize_maps(const std::string &text)
{
	const std::string prefix = "/bpftime-dyld";
	std::string out;
	out.reserve(text.size() + 64);
	size_t pos = 0;
	while (pos < text.size()) {
		size_t eol = text.find('\n', pos);
		if (eol == std::string::npos)
			eol = text.size();
		std::string line = text.substr(pos, eol - pos);
		for (const char *pref : { "/usr/lib/", "/System/Library/" }) {
			size_t p = line.find(pref);
			if (p != std::string::npos) {
				line.insert(p, prefix);
				break;
			}
		}
		out += line;
		out.push_back('\n');
		pos = eol + 1;
	}
	return out;
}

const UmbrellaIndex &libsystem_umbrella()
{
	if (!g_umbrella_built) {
		g_umbrella = build_libsystem_umbrella(kLibSystemVirtual);
		g_umbrella_built = true;
	}
	return g_umbrella;
}

ModuleSymbols resolve_module_symbols(const std::string &path)
{
	std::lock_guard<std::mutex> guard(g_symbol_mutex);
	auto cached = g_symbol_cache.find(path);
	if (cached != g_symbol_cache.end())
		return cached->second;
	ModuleSymbols mod;
	if (is_dyld_virtual_path(path)) {
		std::string real = dyld_virtual_to_real(path);
		if (real == kLibSystemReal) {
			mod = libsystem_umbrella().stub;
		} else {
			mod = enumerate_runtime_exports(real);
			mod.canonical_path = path;
		}
	} else {
		mod = parse_disk_macho(path);
	}
	auto [it, _] = g_symbol_cache.emplace(path, std::move(mod));
	return it->second;
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
	return { "darwin", arch, (int)sysconf(_SC_NPROCESSORS_ONLN) };
}

void platform_init()
{
	struct sigaction sa {};
	sa.sa_handler = reap_children;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_RESTART | SA_NOCLDSTOP;
	sigaction(SIGCHLD, &sa, nullptr);
	// Frida Gum powers dyld_shared_cache export enumeration; initialize its
	// process-global state before any gum_module_* call.
	gum_init_embedded();
}

std::string platform_default_agent_path(const char *argv0)
{
	if (const char *env = getenv("BPFTIME_AGENT_LIB"); env && env[0])
		return env;
	char resolved[4096] = {};
	if (realpath(argv0, resolved) == nullptr)
		return "libbpftime-agent.dylib";
	std::string exe(resolved);
	size_t slash = exe.find_last_of('/');
	std::string dir = slash == std::string::npos ? "." :
						       exe.substr(0, slash);
	return dir + "/../../runtime/agent/libbpftime-agent.dylib";
}

const char *platform_preload_env_name()
{
	return "DYLD_INSERT_LIBRARIES";
}

ProcCheckResult platform_proc_check(pid_t pid)
{
	ProcCheckResult r;
	if (pid <= 0 || kill(pid, 0) != 0)
		return r;
	r.alive = true;
	char path[PROC_PIDPATHINFO_MAXSIZE] = {};
	int n = proc_pidpath(pid, path, sizeof(path));
	if (n > 0) {
		r.exe = path;
		size_t slash = r.exe.find_last_of('/');
		r.comm = slash == std::string::npos ? r.exe :
						      r.exe.substr(slash + 1);
	}
	return r;
}

std::string platform_proc_maps(pid_t pid)
{
	// The injected agent dumps its dyld module snapshot here; absent that
	// file there is no /proc on Darwin to fall back to.
	std::string maps_path =
		"/tmp/bpftime-maps-" + std::to_string(pid) + ".txt";
	std::ifstream f(maps_path);
	if (!f)
		return {};
	std::stringstream ss;
	ss << f.rdbuf();
	return virtualize_maps(ss.str());
}

std::vector<uint8_t> platform_fetch_elf(const std::string &path,
					std::string &canonical_path)
{
	ModuleSymbols mod = resolve_module_symbols(path);
	if (mod.symbols.empty())
		SPDLOG_WARN("fetch_elf: no symbols extracted for {}", path);
	canonical_path = mod.canonical_path.empty() ? path :
						       mod.canonical_path;
	return build_stub_elf(mod);
}

bool platform_translate_uprobe(std::string &module, uint64_t &offset)
{
	if (module == kLibSystemVirtual || module == kLibSystemReal) {
		const UmbrellaIndex &idx = libsystem_umbrella();
		std::string sym_name;
		for (const auto &s : idx.stub.symbols) {
			if (s.offset == offset) {
				sym_name = s.name;
				break;
			}
		}
		auto deep = idx.deep_symbols.find(sym_name);
		if (deep != idx.deep_symbols.end()) {
			module = deep->second.module;
			offset = deep->second.offset;
			return true;
		}
		auto translated = idx.translations.find(offset);
		if (translated != idx.translations.end()) {
			module = translated->second.first;
			offset = translated->second.second;
			return true;
		}
	} else if (is_dyld_virtual_path(module)) {
		module = dyld_virtual_to_real(module);
		return true;
	}
	return false;
}

pid_t platform_spawn(char *const argv[], char *const envp[],
		     const std::string &cwd, bool hold)
{
	posix_spawn_file_actions_t actions;
	posix_spawn_file_actions_init(&actions);
	if (!cwd.empty()) {
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&                               \
	__MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
		posix_spawn_file_actions_addchdir(&actions, cwd.c_str());
#else
		posix_spawn_file_actions_addchdir_np(&actions, cwd.c_str());
#endif
	}
	posix_spawnattr_t attr;
	posix_spawnattr_init(&attr);
	if (hold)
		posix_spawnattr_setflags(&attr, POSIX_SPAWN_START_SUSPENDED);

	pid_t pid = -1;
	int rc = posix_spawn(&pid, argv[0], &actions, &attr, argv, envp);
	posix_spawnattr_destroy(&attr);
	posix_spawn_file_actions_destroy(&actions);
	if (rc != 0) {
		errno = rc;
		throw std::runtime_error(std::string("posix_spawn failed: ") +
					 strerror(rc));
	}
	if (hold)
		SPDLOG_DEBUG("spawn: pid {} suspended awaiting resume", pid);
	return pid;
}

void platform_resume(pid_t pid)
{
	if (pid > 0)
		kill(pid, SIGCONT);
}

void platform_refresh_agent(pid_t)
{
	// The agent IPC socket is Linux-only. On Darwin, probes created while a
	// POSIX_SPAWN_START_SUSPENDED child is held are attached when the agent
	// constructor runs after SIGCONT, so no explicit refresh is needed.
}

} // namespace remote
} // namespace bpftime

#endif // __APPLE__
