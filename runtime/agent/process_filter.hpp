#ifndef BPFTIME_AGENT_PROCESS_FILTER_HPP
#define BPFTIME_AGENT_PROCESS_FILTER_HPP
// Stack-only early admission filter (no heap/STL): runs before logger/SHM/frida init.
#include <cerrno>
#include <cstdlib>
#include <cstring>
#if defined(__linux__)
#include <unistd.h>
#if !defined(_GNU_SOURCE)
extern "C" char *program_invocation_short_name; // exported by glibc/musl even without _GNU_SOURCE.
#endif
#endif
namespace bpftime::agent {
// Match name against a comma-separated list; tokens trim spaces/tabs, empty tokens ignored.
inline bool process_name_in_list(const char *name, const char *list) noexcept
{
	if (!name || !*name || !list) return false;
	size_t name_len = strlen(name);
	for (const char *p = list;;) {
		while (*p == ' ' || *p == '\t' || *p == ',') ++p;
		const char *token = p;
		while (*p != '\0' && *p != ',') ++p;
		const char *end = p;
		while (end > token && (end[-1] == ' ' || end[-1] == '\t')) --end;
		size_t len = static_cast<size_t>(end - token);
		if (len == name_len && memcmp(token, name, len) == 0) return true;
		if (*p == '\0') return false;
	}
}
// Linux readlink appends " (deleted)" when the executable was unlinked/replaced.
inline void strip_deleted_marker(char *buf) noexcept
{
	constexpr char marker[] = " (deleted)";
	size_t len = strlen(buf), mlen = sizeof(marker) - 1;
	if (len >= mlen && memcmp(buf + len - mlen, marker, mlen) == 0)
		buf[len - mlen] = '\0';
}
// DENY wins; non-empty ALLOW needs an argv[0]/exe basename match; empty lists allow all.
inline bool process_filter_skips(const char *deny, const char *allow,
				 const char *argv0_name,
				 const char *exe_name) noexcept
{
	if (deny && *deny && (process_name_in_list(argv0_name, deny) ||
			      process_name_in_list(exe_name, deny)))
		return true;
	if (allow && *allow && !process_name_in_list(argv0_name, allow) &&
	    !process_name_in_list(exe_name, allow))
		return true;
	return false;
}
// Read-only env access keeps LD_PRELOAD intact for children. argv0_override is the
// argv[0] basename from __libc_start_main; musl initializes its program-name symbol
// only inside that entry, so the interposer must supply argv[0] itself.
inline bool should_skip_agent_process(const char *argv0_override = nullptr) noexcept
{
	int saved_errno = errno;
	const char *deny = getenv("BPFTIME_AGENT_DENY_PROCESS");
	const char *allow = getenv("BPFTIME_AGENT_ALLOW_PROCESS");
	if ((deny == nullptr || *deny == '\0') && (allow == nullptr || *allow == '\0')) {
		errno = saved_errno;
		return false;
	}
	const char *argv0_name = argv0_override;
#if defined(__linux__)
	if (!argv0_name)
		argv0_name = program_invocation_short_name;
	char exe_buf[4096];
	ssize_t n = ::readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
	const char *exe_name = nullptr;
	if (n > 0) {
		exe_buf[n] = '\0';
		strip_deleted_marker(exe_buf);
		exe_name = strrchr(exe_buf, '/') + 1; // /proc/self/exe is always absolute.
	}
#elif defined(__APPLE__)
	if (!argv0_name)
		argv0_name = getprogname();
	const char *exe_name = nullptr;
#else
	const char *exe_name = nullptr;
#endif
	bool skip = process_filter_skips(deny, allow, argv0_name, exe_name);
	errno = saved_errno;
	return skip;
}
} // namespace bpftime::agent
#endif
