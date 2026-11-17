/* SPDX-License-Identifier: MIT
 *
 * Built-in synthetic /proc/kallsyms content for rootless operation.
 *
 * In an unprivileged container the kernel hides symbol addresses:
 * /proc/kallsyms reports every address as zero, which makes strict parsers
 * (e.g. the OpenTelemetry eBPF profiler's NewSymbolizer) fail with an
 * ErrSymbolPermissions-style error before any eBPF program is loaded.
 *
 * The real kernel addresses are never dereferenced under bpftime:
 *  - bpf_get_current_task() returns a synthetic per-thread task buffer,
 *  - system-analysis programs are completed from map values instead of
 *    reading kernel memory, and
 *  - bpf_probe_read_kernel() on an unreachable address fails with -EFAULT
 *    rather than leaking kernel memory.
 *
 * Providing a stable, well-formed table therefore lets an unmodified
 * userspace profiler initialize without a real System.map, while keeping
 * every address inside a non-canonical/high range that userspace cannot
 * map, so an accidental direct access faults safely.
 */
#ifndef BPFTIME_SYNTHETIC_KALLSYMS_H
#define BPFTIME_SYNTHETIC_KALLSYMS_H

#include <string>
#include <string_view>

namespace bpftime
{

// Returns the synthetic /proc/kallsyms body. Entries follow the kernel
// format: "<hex address> <type> <name>\n". All addresses are nonzero text
// symbols so parsers that require at least one real address accept it.
//
// The set covers the symbols a generic eBPF tracer looks up during
// initialization (module boundaries and a handful of well-known functions).
// Unknown additional symbols simply fail lookup and are handled by callers.
inline const std::string &synthetic_kallsyms_content()
{
	// Addresses live in a high canonical-kernel-style range. They are
	// deliberately not real offsets of any running kernel.
	static const std::string content =
		"ffffffff81000000 T _stext\n"
		"ffffffff81000100 T _text\n"
		"ffffffff81001000 T startup_64\n"
		"ffffffff81200000 T copy_from_user_nofault\n"
		"ffffffff81200100 T nmi_uaccess_okay\n"
		"ffffffff81200200 T finish_task_switch\n"
		"ffffffff81200300 T __schedule\n"
		"ffffffff81200400 T sched_switch\n"
		"ffffffff81200500 T __x64_sys_getpid\n"
		"ffffffff81200600 T prctl_set_seccomp\n"
		"ffffffff82000000 T _etext\n";
	return content;
}

// True for the one path the synthetic table can replace.
inline bool is_synthetic_kallsyms_path(const std::string &path)
{
	return path == "/proc/kallsyms";
}

// True when a sample of /proc/kallsyms carries no nonzero address, which is
// what kptr_restrict without CAP_SYSLOG produces in an unprivileged
// container. A privileged kernel reports real addresses, and then the real
// file has to be served: fabricated addresses would make every
// address-to-symbol lookup in the traced process wrong, and their static
// offsets do not even match the running kernel.
inline bool kallsyms_addresses_hidden(std::string_view sample)
{
	size_t pos = 0;
	while (pos < sample.size()) {
		size_t end = sample.find('\n', pos);
		if (end == std::string_view::npos)
			end = sample.size();
		std::string_view line = sample.substr(pos, end - pos);
		pos = end + 1;
		size_t space = line.find(' ');
		if (space == std::string_view::npos || space == 0)
			continue;
		for (char ch : line.substr(0, space)) {
			if (ch != '0')
				return false;
		}
	}
	return true;
}

} // namespace bpftime

#endif // BPFTIME_SYNTHETIC_KALLSYMS_H