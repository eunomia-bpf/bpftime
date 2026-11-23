/* SPDX-License-Identifier: MIT */
#include "rootless_map_files.hpp"
#include "seccomp_raw_syscall.hpp"
#include <cerrno>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#include <sys/syscall.h>

namespace bpftime
{
namespace rootless
{

namespace
{

int raw_openat(const char *path, int flags, mode_t mode)
{
	return (int)seccomp::raw_syscall(__NR_openat, AT_FDCWD,
					 (intptr_t)path, flags, (intptr_t)mode,
					 0, 0);
}

int raw_fstatat(const char *path, struct stat *st)
{
	return (int)seccomp::raw_syscall(__NR_newfstatat, AT_FDCWD,
					 (intptr_t)path, (intptr_t)st, 0, 0, 0);
}

std::string unescape_proc_path(std::string path)
{
	std::string result;
	result.reserve(path.size());
	for (size_t i = 0; i < path.size(); i++) {
		if (path[i] == '\\' && i + 3 < path.size() &&
		    path[i + 1] >= '0' && path[i + 1] <= '7' &&
		    path[i + 2] >= '0' && path[i + 2] <= '7' &&
		    path[i + 3] >= '0' && path[i + 3] <= '7') {
			result.push_back((char)((path[i + 1] - '0') * 64 +
						(path[i + 2] - '0') * 8 +
						path[i + 3] - '0'));
			i += 3;
		} else {
			result.push_back(path[i]);
		}
	}
	return result;
}

} // namespace

bool parse_map_files_path(const char *path, pid_t &pid,
			  unsigned long long &start,
			  unsigned long long &end)
{
	if (path == nullptr)
		return false;
	int consumed = 0;
	unsigned parsed_pid = 0;
	if (sscanf(path, "/proc/%u/map_files/%llx-%llx%n", &parsed_pid,
		   &start, &end, &consumed) != 3 ||
	    path[consumed] != '\0' || parsed_pid == 0 || start >= end)
		return false;
	pid = (pid_t)parsed_pid;
	return true;
}

bool is_map_files_path(const char *path)
{
	pid_t pid;
	unsigned long long start, end;
	return parse_map_files_path(path, pid, start, end);
}

namespace detail
{
std::optional<std::string>
select_accessible_path(const std::string &logical_path,
		       const std::string &proc_root_path, int root_stat_rc,
		       const struct stat *root_st, int logical_stat_rc,
		       const struct stat *logical_st,
		       unsigned long long maps_inode, dev_t maps_dev)
{
	auto matches_mapping = [&](const struct stat &st) {
		return (maps_inode == 0 || st.st_ino == (ino_t)maps_inode) &&
		       st.st_dev == maps_dev;
	};
	if (root_stat_rc == 0 && root_st != nullptr &&
	    matches_mapping(*root_st))
		return proc_root_path;
	// /proc/<pid>/root is gated by ptrace access (yama ptrace_scope >= 2 or
	// a non-dumpable target). On denial, a same-mount-namespace process can
	// open the logical path directly; accept it only when dev+inode prove
	// it names the same backing inode as the mapping.
	if ((root_stat_rc == -EACCES || root_stat_rc == -EPERM) &&
	    logical_stat_rc == 0 && logical_st != nullptr &&
	    matches_mapping(*logical_st))
		return logical_path;
	return {};
}
} // namespace detail

std::optional<mapped_file_resolution>
resolve_map_files_path(const char *path)
{
	pid_t pid;
	unsigned long long wanted_start, wanted_end;
	if (!parse_map_files_path(path, pid, wanted_start, wanted_end))
		return {};

	std::string maps_path = "/proc/" + std::to_string(pid) + "/maps";
	int maps_fd = raw_openat(maps_path.c_str(), O_RDONLY | O_CLOEXEC, 0);
	if (maps_fd < 0)
		return {};
	// fdopen is not interposed by this object; the stream wraps the raw
	// fd so getline() never reaches the interposed fopen symbol either.
	FILE *maps = fdopen(maps_fd, "re");
	if (maps == nullptr) {
		seccomp::raw_syscall(__NR_close, maps_fd, 0, 0, 0, 0, 0);
		return {};
	}

	std::optional<mapped_file_resolution> result;
	char *line = nullptr;
	size_t capacity = 0;
	while (getline(&line, &capacity, maps) >= 0) {
		unsigned long long start, end, offset, inode;
		unsigned dev_major, dev_minor;
		char permissions[5] = {};
		int path_offset = 0;
		if (sscanf(line, "%llx-%llx %4s %llx %x:%x %llu %n", &start,
			   &end, permissions, &offset, &dev_major, &dev_minor,
			   &inode, &path_offset) != 7 ||
		    start != wanted_start || end != wanted_end)
			continue;
		std::string logical(line + path_offset);
		while (!logical.empty() &&
		       (logical.back() == '\n' || logical.back() == '\r'))
			logical.pop_back();
		logical = unescape_proc_path(logical);
		if (logical.empty() || logical.front() != '/' ||
		    logical.ends_with(" (deleted)"))
			break;
		std::string proc_root = "/proc/" + std::to_string(pid) +
					"/root" + logical;
		struct stat root_st = {};
		int root_rc = raw_fstatat(proc_root.c_str(), &root_st);
		struct stat logical_st = {};
		int logical_rc = -1;
		if (root_rc == -EACCES || root_rc == -EPERM)
			logical_rc = raw_fstatat(logical.c_str(), &logical_st);
		auto accessible = detail::select_accessible_path(
			logical, proc_root, root_rc, &root_st, logical_rc,
			logical_rc == 0 ? &logical_st : nullptr, inode,
			makedev(dev_major, dev_minor));
		if (accessible)
			result = mapped_file_resolution{
				std::move(logical), std::move(*accessible) };
		break;
	}
	free(line);
	fclose(maps);
	return result;
}

int open_map_files(const char *path, int oflag, mode_t mode)
{
	auto resolution = resolve_map_files_path(path);
	if (!resolution) {
		errno = EACCES;
		return -1;
	}
	int64_t rc = seccomp::raw_syscall(
		__NR_openat, AT_FDCWD,
		(intptr_t)resolution->accessible_path.c_str(), oflag,
		(intptr_t)mode, 0, 0);
	if (rc < 0) {
		errno = (int)-rc;
		return -1;
	}
	return (int)rc;
}

FILE *fopen_map_files(const char *path, const char *flags)
{
	if (flags == nullptr) {
		errno = EINVAL;
		return nullptr;
	}
	bool update = strchr(flags, '+') != nullptr;
	int oflag = O_RDONLY;
	if (strchr(flags, 'w') != nullptr)
		oflag = (update ? O_RDWR : O_WRONLY) | O_CREAT | O_TRUNC;
	else if (strchr(flags, 'a') != nullptr)
		oflag = (update ? O_RDWR : O_WRONLY) | O_CREAT | O_APPEND;
	else if (update)
		oflag = O_RDWR;
	if (strchr(flags, 'e') != nullptr)
		oflag |= O_CLOEXEC;
	int fd = open_map_files(path, oflag, 0666);
	if (fd < 0)
		return nullptr;
	FILE *fp = fdopen(fd, flags);
	if (fp == nullptr) {
		int saved = errno;
		seccomp::raw_syscall(__NR_close, fd, 0, 0, 0, 0, 0);
		errno = saved;
	}
	return fp;
}

} // namespace rootless
} // namespace bpftime

#else // Non-Linux builds keep linkable stubs; map_files is Linux-only.

namespace bpftime
{
namespace rootless
{

bool parse_map_files_path(const char *, pid_t &, unsigned long long &,
			  unsigned long long &)
{
	return false;
}

bool is_map_files_path(const char *)
{
	return false;
}

std::optional<mapped_file_resolution>
resolve_map_files_path(const char *)
{
	return {};
}

int open_map_files(const char *, int, mode_t)
{
	errno = ENOSYS;
	return -1;
}

FILE *fopen_map_files(const char *, const char *)
{
	errno = ENOSYS;
	return nullptr;
}

namespace detail
{
std::optional<std::string>
select_accessible_path(const std::string &, const std::string &, int,
		       const struct stat *, int, const struct stat *,
		       unsigned long long, dev_t)
{
	return {};
}
} // namespace detail

} // namespace rootless
} // namespace bpftime

#endif
