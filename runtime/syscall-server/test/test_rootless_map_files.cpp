// Rootless /proc/<pid>/map_files resolution tests.
//
// Two layers are covered:
//   1. The pure candidate-selection logic (detail::select_accessible_path),
//      including the simulated /proc/<pid>/root EACCES/EPERM fallback and its
//      st_dev + st_ino validation. A real kernel cannot produce
//      "maps readable but /proc/<pid>/root denied" on demand (that needs
//      yama.ptrace_scope >= 2, which is not writable inside containers), so
//      the decision function is exercised with synthetic stat results.
//   2. A live /proc/<pid>/map_files/<start>-<end> range in an unprivileged
//      process (no effective capabilities): the direct map_files open is
//      expected to fail with EPERM, and the helper must still resolve, open
//      and read the backing file with no extra capability.

#include "../rootless_map_files.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cerrno>
#include <cstdio>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using bpftime::rootless::detail::select_accessible_path;

namespace
{
constexpr dev_t kMapsDev = 0x10303;
constexpr unsigned long long kMapsInode = 424242;
const std::string kLogical = "/tmp/bpftime-rootless-map-files-test";
const std::string kProcRoot =
	"/proc/self/root/tmp/bpftime-rootless-map-files-test";

struct stat make_stat(dev_t dev, ino_t ino)
{
	struct stat st = {};
	st.st_dev = dev;
	st.st_ino = ino;
	return st;
}
} // namespace

TEST_CASE("map_files path parsing accepts only exact links", "[rootless]")
{
	pid_t pid = 0;
	unsigned long long start = 0, end = 0;
	REQUIRE(bpftime::rootless::parse_map_files_path(
			"/proc/1234/map_files/7f0000-7f1000", pid, start,
			end));
	REQUIRE(pid == 1234);
	REQUIRE(start == 0x7f0000ULL);
	REQUIRE(end == 0x7f1000ULL);
	REQUIRE(bpftime::rootless::is_map_files_path(
		"/proc/1/map_files/0-1"));

	const char *const invalid[] = {
		nullptr,
		"",
		"/proc/0/map_files/1-2",
		"/proc/1234/map_files/7f1000-7f0000",
		"/proc/1234/map_files/7f0000-7f0000",
		"/proc/1234/map_files/7f0000-7f1000/trailer",
		"/proc/abc/map_files/7f0000-7f1000",
		"/proc/1234/maps/7f0000-7f1000",
	};
	for (const char *path : invalid) {
		REQUIRE_FALSE(bpftime::rootless::is_map_files_path(path));
	}
}

TEST_CASE("validated /proc/<pid>/root path is preferred", "[rootless]")
{
	struct stat root_st = make_stat(kMapsDev, (ino_t)kMapsInode);
	auto picked = select_accessible_path(
		kLogical, kProcRoot, 0, &root_st, -1, nullptr, kMapsInode,
		kMapsDev);
	REQUIRE(picked.has_value());
	REQUIRE(*picked == kProcRoot);
}

TEST_CASE("root EACCES falls back to the logical path on dev+inode match",
	  "[rootless]")
{
	struct stat logical_st = make_stat(kMapsDev, (ino_t)kMapsInode);
	for (int denial : { -EACCES, -EPERM }) {
		auto picked = select_accessible_path(
			kLogical, kProcRoot, denial, nullptr, 0, &logical_st,
			kMapsInode, kMapsDev);
		REQUIRE(picked.has_value());
		REQUIRE(*picked == kLogical);
	}
}

TEST_CASE("logical fallback is rejected on st_ino mismatch", "[rootless]")
{
	struct stat wrong_inode = make_stat(kMapsDev, (ino_t)kMapsInode + 1);
	auto picked = select_accessible_path(
		kLogical, kProcRoot, -EACCES, nullptr, 0, &wrong_inode,
		kMapsInode, kMapsDev);
	REQUIRE_FALSE(picked.has_value());
}

TEST_CASE("st_dev mismatch with matching inode is accepted (overlayfs)",
	  "[rootless]")
{
	// overlayfs/nydus: the live mount reports a different st_dev than the
	// backing lower-fs dev recorded in /proc/<pid>/maps for the same inode.
	// The inode is the authoritative identity for a file-backed mapping.
	struct stat wrong_dev =
		make_stat(kMapsDev + 1, (ino_t)kMapsInode);
	auto picked = select_accessible_path(kLogical, kProcRoot, -EACCES,
						nullptr, 0, &wrong_dev,
						kMapsInode, kMapsDev);
	REQUIRE(picked.has_value());
	REQUIRE(*picked == kLogical);

	// Same divergence on an accessible /proc/<pid>/root view resolves to
	// the root path without needing the logical fallback.
	picked = select_accessible_path(kLogical, kProcRoot, 0, &wrong_dev, -1,
					nullptr, kMapsInode, kMapsDev);
	REQUIRE(picked.has_value());
	REQUIRE(*picked == kProcRoot);
}

TEST_CASE("non-denial root errors never trigger the logical fallback",
	  "[rootless]")
{
	struct stat logical_st = make_stat(kMapsDev, (ino_t)kMapsInode);
	auto picked = select_accessible_path(
		kLogical, kProcRoot, -ENOENT, nullptr, 0, &logical_st,
		kMapsInode, kMapsDev);
	REQUIRE_FALSE(picked.has_value());
}

TEST_CASE("inode-zero mappings still require an st_dev match", "[rootless]")
{
	struct stat root_st = make_stat(kMapsDev, 999);
	auto picked = select_accessible_path(
		kLogical, kProcRoot, 0, &root_st, -1, nullptr, 0, kMapsDev);
	REQUIRE(picked.has_value());
	REQUIRE(*picked == kProcRoot);

	struct stat other_dev = make_stat(kMapsDev + 1, 999);
	picked = select_accessible_path(kLogical, kProcRoot, -EPERM, nullptr,
					0, &other_dev, 0, kMapsDev);
	REQUIRE_FALSE(picked.has_value());
}

TEST_CASE("live unprivileged map_files range resolves and opens", "[rootless]")
{
	char tmp_path[] = "/tmp/bpftime-mf-live.XXXXXX";
	int tmp_fd = mkstemp(tmp_path);
	REQUIRE(tmp_fd >= 0);
	const char payload[] = "rootless map_files live payload\n";
	REQUIRE(write(tmp_fd, payload, sizeof(payload) - 1) ==
		sizeof(payload) - 1);

	struct stat expected = {};
	REQUIRE(fstat(tmp_fd, &expected) == 0);

	void *mapping = mmap(nullptr, 4096, PROT_READ, MAP_PRIVATE, tmp_fd,
			     0);
	REQUIRE(mapping != MAP_FAILED);
	auto map_start = (unsigned long long)mapping;
	// The resolver deliberately accepts numeric pids only (kernel and uprobe
	// paths never use the "self" alias), so build the link from getpid();
	// map_files names encode the range in hexadecimal.
	char map_files_buf[128];
	snprintf(map_files_buf, sizeof(map_files_buf),
		 "/proc/%d/map_files/%llx-%llx", (int)getpid(), map_start,
		 map_start + 4096);
	std::string map_files(map_files_buf);

	SECTION("resolution validates the backing file against maps metadata")
	{
		auto resolution = bpftime::rootless::resolve_map_files_path(
			map_files.c_str());
		REQUIRE(resolution.has_value());
		REQUIRE(resolution->logical_path == tmp_path);

		struct stat accessible = {};
		REQUIRE(stat(resolution->accessible_path.c_str(),
			     &accessible) == 0);
		REQUIRE(accessible.st_ino == expected.st_ino);
		REQUIRE(accessible.st_dev == expected.st_dev);
	}

	SECTION("open_map_files succeeds after the EPERM-denied direct open")
	{
		errno = 0;
		int direct = open(map_files.c_str(), O_RDONLY);
		if (direct < 0) {
			// Unprivileged container: map_files opens require
			// CAP_SYS_ADMIN, this is the production failure mode.
			REQUIRE((errno == EPERM || errno == EACCES));
		} else {
			close(direct);
		}

		int fd = bpftime::rootless::open_map_files(
			map_files.c_str(), O_RDONLY, 0);
		REQUIRE(fd >= 0);
		char buffer[sizeof(payload)] = {};
		REQUIRE(read(fd, buffer, sizeof(payload) - 1) ==
			sizeof(payload) - 1);
		REQUIRE(std::string(buffer) == payload);
		close(fd);

		FILE *fp = bpftime::rootless::fopen_map_files(
			map_files.c_str(), "re");
		REQUIRE(fp != nullptr);
		char line[sizeof(payload)] = {};
		REQUIRE(fgets(line, sizeof(line), fp) != nullptr);
		REQUIRE(std::string(line) == payload);
		fclose(fp);
	}

	SECTION("non-map_files and unknown targets resolve to nothing")
	{
		REQUIRE_FALSE(bpftime::rootless::resolve_map_files_path(
			"/etc/hostname"));
		REQUIRE_FALSE(bpftime::rootless::resolve_map_files_path(
			"/proc/999999/map_files/1-2"));
	}

	munmap(mapping, 4096);
	close(tmp_fd);
	unlink(tmp_path);
}
