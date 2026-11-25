/* SPDX-License-Identifier: MIT
 *
 * Shared rootless resolver for /proc/<pid>/map_files/<start>-<end> links.
 *
 * Opening a map_files link requires CAP_SYS_ADMIN (or CAP_CHECKPOINT_RESTORE),
 * so an unprivileged process gets EPERM even for its own mappings. The same
 * information is available through /proc/<pid>/maps, which remains readable
 * for same-UID processes. This helper decodes the logical backing file there
 * and validates it against the mapping's st_dev/st_ino before returning a
 * path that can be opened without any extra capability.
 *
 * Every file operation below issues raw syscalls (see seccomp_raw_syscall.hpp)
 * so the resolver never re-enters the open/openat/fopen symbols this shared
 * object interposes.
 */
#ifndef BPFTIME_ROOTLESS_MAP_FILES_HPP
#define BPFTIME_ROOTLESS_MAP_FILES_HPP

#include <cstdio>
#include <optional>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

namespace bpftime
{
namespace rootless
{

struct mapped_file_resolution {
	// Backing file path decoded from the /proc/<pid>/maps line, i.e. the
	// map_files readlink target.
	std::string logical_path;
	// Path that is actually openable by this process: normally
	// /proc/<pid>/root + logical_path, or logical_path itself when
	// /proc/<pid>/root is denied by ptrace permissions and the mapping
	// lives in the same mount namespace.
	std::string accessible_path;
};

// Parse "/proc/<pid>/map_files/<start>-<end>" with no trailing characters.
bool parse_map_files_path(const char *path, pid_t &pid,
			  unsigned long long &start,
			  unsigned long long &end);

// Convenience boolean wrapper around parse_map_files_path().
bool is_map_files_path(const char *path);

// Resolve a map_files link to an openable backing file. Returns std::nullopt
// for non-map_files paths, unreadable /proc/<pid>/maps or mappings that have
// no validated file backing.
std::optional<mapped_file_resolution>
resolve_map_files_path(const char *path);

// Open a map_files path after a direct kernel open failed with EPERM/EACCES.
// Returns a kernel fd, or -1 with errno set when no validated backing file
// exists.
int open_map_files(const char *path, int oflag, mode_t mode);

// fopen(3)-style wrapper around open_map_files(); the returned stream owns a
// freshly opened kernel fd.
FILE *fopen_map_files(const char *path, const char *flags);

namespace detail
{
// Pure candidate selection, exposed so the /proc/<pid>/root denial fallback
// and the st_dev/st_ino validation are testable without procfs permission
// control. Stat results use kernel-style negative errno values (e.g.
// -EACCES); nullptr stat structs are treated as a failed stat.
std::optional<std::string>
select_accessible_path(const std::string &logical_path,
		       const std::string &proc_root_path, int root_stat_rc,
		       const struct stat *root_st, int logical_stat_rc,
		       const struct stat *logical_st,
		       unsigned long long maps_inode, dev_t maps_dev);
} // namespace detail

} // namespace rootless
} // namespace bpftime

#endif
