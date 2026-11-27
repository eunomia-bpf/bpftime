/* SPDX-License-Identifier: MIT
 *
 * OS-specific backend contract for bpftime-remote-host. Exactly one
 * implementation is compiled: linux_target.cpp on __linux__ and
 * darwin_target.cpp (+ macho_reader.cpp / elf_synth.cpp) on __APPLE__.
 */
#ifndef BPFTIME_REMOTE_PLATFORM_TARGET_HPP
#define BPFTIME_REMOTE_PLATFORM_TARGET_HPP

#include <cstdint>
#include <string>
#include <sys/types.h>
#include <vector>

namespace bpftime
{
namespace remote
{

struct PlatformIdentity {
	const char *os; // "linux" | "darwin"
	const char *arch; // "arm64" | "x86_64"
	int num_cpus;
};

PlatformIdentity platform_identity();

// One-time process setup (Frida Gum init on Darwin, no-op on Linux).
void platform_init();

// Absolute path to the agent shared library co-located with this daemon,
// unless BPFTIME_AGENT_LIB overrides it.
std::string platform_default_agent_path(const char *argv0);

// Dynamic-loader injection variable: LD_PRELOAD (Linux) or
// DYLD_INSERT_LIBRARIES (Darwin). The daemon assigns it the absolute agent
// library path.
const char *platform_preload_env_name();

struct ProcCheckResult {
	bool alive = false;
	std::string comm;
	std::string exe;
	std::string status;
};

ProcCheckResult platform_proc_check(pid_t pid);

// Linux /proc/<pid>/maps text, or a synthesized equivalent on Darwin.
std::string platform_proc_maps(pid_t pid);

// Resolve a target module path to raw 64-bit ELF bytes. Linux returns the
// real on-disk ELF; Darwin parses Mach-O and synthesizes a stub ELF.
// `canonical_path` is set to the normalized module path.
std::vector<uint8_t> platform_fetch_elf(const std::string &path,
					std::string &canonical_path);

// Darwin-only: translate a synthetic libSystem umbrella / dyld virtual uprobe
// target to the deep hookable module and offset. Linux leaves it unchanged and
// returns false.
bool platform_translate_uprobe(std::string &module, uint64_t &offset);

// Spawn a workload with the already-built argv/envp. When hold is true the
// child is stopped before its entry code runs and the call blocks until that
// stopped state is observable. Returns the child pid. Throws on failure.
pid_t platform_spawn(char *const argv[], char *const envp[],
		     const std::string &cwd, bool hold);

// Continue a child previously spawned with hold=true.
void platform_resume(pid_t pid);

// Best-effort incremental (-p) attach trigger for a running agent.
void platform_refresh_agent(pid_t pid);

} // namespace remote
} // namespace bpftime

#endif
