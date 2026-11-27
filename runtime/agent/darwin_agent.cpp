/* SPDX-License-Identifier: MIT
 *
 * Darwin (Apple Silicon) agent support: dyld image enumeration and the
 * one-shot shared-memory readiness probe. No FIFO, no polling, no blocking:
 * the spawner guarantees SHM readiness before it resumes the suspended child.
 */
#ifdef __APPLE__
#include "darwin_agent.hpp"
#include "handler/handler_manager.hpp"

#include <cerrno>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <mach/mach.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <spdlog/spdlog.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace bpftime
{
namespace agent_darwin
{

namespace
{
char prot_char(vm_prot_t prot, vm_prot_t bit, char letter)
{
	return (prot & bit) ? letter : '-';
}

uint64_t synthetic_image_inode(const char *image_name)
{
	if (image_name == nullptr || image_name[0] == '\0')
		return 0;
	uint32_t h = 2166136261U;
	for (const unsigned char *p =
		     reinterpret_cast<const unsigned char *>(image_name);
	     *p != '\0'; ++p) {
		h ^= *p;
		h *= 16777619U;
	}
	return 100000ULL + static_cast<uint64_t>(h);
}

// Walk one loaded image's load commands and append one Linux maps style line
// per segment, all carrying the image path. The __TEXT segment start
// (slide + vmaddr) is the module base used for uprobe offset translation.
void append_image_lines(const mach_header *mh, intptr_t slide,
			const char *image_name, std::string &out)
{
	if (mh->magic != MH_MAGIC_64)
		return;
	auto *header = reinterpret_cast<const mach_header_64 *>(mh);
	const auto *first_cmd = reinterpret_cast<const load_command *>(
		reinterpret_cast<const uint8_t *>(header) +
		sizeof(mach_header_64));
	uint64_t text_vmaddr = 0;
	bool have_text_vmaddr = false;
	const load_command *scan = first_cmd;
	for (uint32_t i = 0; i < header->ncmds;
	     i++, scan = reinterpret_cast<const load_command *>(
			  reinterpret_cast<const uint8_t *>(scan) +
			  scan->cmdsize)) {
		if (scan->cmd != LC_SEGMENT_64)
			continue;
		const auto *seg =
			reinterpret_cast<const segment_command_64 *>(scan);
		if (strcmp(seg->segname, "__TEXT") == 0) {
			text_vmaddr = seg->vmaddr;
			have_text_vmaddr = true;
			break;
		}
	}

	uint64_t inode = synthetic_image_inode(image_name);
	const load_command *cmd = first_cmd;
	for (uint32_t i = 0; i < header->ncmds;
	     i++, cmd = reinterpret_cast<const load_command *>(
			  reinterpret_cast<const uint8_t *>(cmd) +
			  cmd->cmdsize)) {
		if (cmd->cmd != LC_SEGMENT_64)
			continue;
		auto *seg = reinterpret_cast<const segment_command_64 *>(cmd);
		if (seg->vmsize == 0 || seg->initprot == 0)
			continue;
		uint64_t start = (uint64_t)slide + (uint64_t)seg->vmaddr;
		uint64_t end = start + seg->vmsize;
		if ((start & 0xfffULL) == 0 &&
		    (seg->initprot & VM_PROT_EXECUTE) != 0)
			end = (end + 0xfffULL) & ~0xfffULL;
		uint64_t rel_off = (have_text_vmaddr &&
				    seg->vmaddr >= text_vmaddr) ?
					   (seg->vmaddr - text_vmaddr) :
					   seg->fileoff;
		char perms[5] = {
			prot_char(seg->initprot, VM_PROT_READ, 'r'),
			prot_char(seg->initprot, VM_PROT_WRITE, 'w'),
			prot_char(seg->initprot, VM_PROT_EXECUTE, 'x'),
			'p', '\0'
		};
		char line[1280];
		int n = snprintf(line, sizeof(line),
				 "%016llx-%016llx %4s %08llx 01:01 %llu %s\n",
				 start, end, perms,
				 (unsigned long long)rel_off,
				 (unsigned long long)inode, image_name);
		if (n > 0)
			out.append(line, (size_t)n);
	}
}

void image_added_cb(const mach_header *mh, intptr_t slide)
{
	(void)mh;
	(void)slide;
	try {
		write_process_maps_file();
	} catch (...) {
		// Never let dyld callback failures take down the host process.
	}
}
} // namespace

std::string maps_file_path(int pid)
{
	return "/tmp/bpftime-maps-" + std::to_string(pid) + ".txt";
}

void write_process_maps_file()
{
	std::string content;
	content.reserve(4096);
	uint32_t count = _dyld_image_count();
	for (uint32_t i = 0; i < count; i++) {
		const char *name = _dyld_get_image_name(i);
		const mach_header *mh = _dyld_get_image_header(i);
		intptr_t slide = _dyld_get_image_vmaddr_slide(i);
		if (name == nullptr || mh == nullptr)
			continue;
		// Skip the injected agent dylib and transitive Apple GUI/private
		// frameworks pulled in by Frida's Objective-C runtime so the
		// maps snapshot reflects the target workload and libSystem
		// modules only.
		if (strstr(name, "libbpftime-agent.dylib") != nullptr ||
		    strncmp(name, "/System/Library/PrivateFrameworks/", 34) ==
			    0 ||
		    strncmp(name, "/System/Library/SubFrameworks/", 30) == 0 ||
		    strncmp(name, "/System/Library/Frameworks/", 27) == 0 ||
		    strncmp(name, "/usr/lib/swift/", 15) == 0 ||
		    strstr(name, "/libswift") != nullptr ||
		    strstr(name, "/libicucore.") != nullptr ||
		    strstr(name, "/libsqlite3.") != nullptr ||
		    strstr(name, "/libxml2.") != nullptr ||
		    strstr(name, "/libarchive.") != nullptr)
			continue;
		append_image_lines(mh, slide, name, content);
	}

	int pid = (int)getpid();
	std::string final_path = maps_file_path(pid);
	std::string tmp_path = final_path + ".tmp." + std::to_string(pid);
	// Exclusive create refuses to follow a pre-existing symlink or open a
	// FIFO at this predictable path; O_CLOEXEC avoids leaking the fd across
	// exec. Any failure fails open: the maps snapshot is best-effort.
	int fd = open(tmp_path.c_str(),
		      O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
	if (fd < 0) {
		SPDLOG_DEBUG("darwin agent: cannot open maps tmp file: {}",
			     strerror(errno));
		return;
	}
	const char *p = content.data();
	size_t left = content.size();
	while (left > 0) {
		ssize_t n = write(fd, p, left);
		if (n <= 0) {
			if (errno == EINTR)
				continue;
			close(fd);
			unlink(tmp_path.c_str());
			return;
		}
		p += n;
		left -= (size_t)n;
	}
	close(fd);
	if (rename(tmp_path.c_str(), final_path.c_str()) != 0) {
		SPDLOG_DEBUG("darwin agent: rename maps file failed: {}",
			     strerror(errno));
		unlink(tmp_path.c_str());
	}
}

void register_image_add_callback()
{
	_dyld_register_func_for_add_image(image_added_cb);
}

bool global_shm_ready_once()
{
	// On Darwin Boost.Interprocess backs shared memory with a filesystem
	// object at /tmp/boost_interprocess/<name> (verified against the Boost
	// version in use), not a native POSIX shm object. Probe it with a
	// stack-only access(2): no heap allocation and no directory creation,
	// unlike shared_memory_object(open_only). Preserve host errno and
	// treat anything other than a present file as "not ready".
	const int saved_errno = errno;
	// Sized for the 23-byte prefix, the maximum 255-byte SHM name, and the
	// terminator (279 bytes), so a supported name is never falsely treated
	// as "not ready" because the pathname did not fit.
	char path[320];
	const char *name = get_global_shm_name();
	bool ready = false;
	if (name != nullptr && name[0] != '\0') {
		int n = snprintf(path, sizeof(path),
				 "/tmp/boost_interprocess/%s", name);
		if (n > 0 && static_cast<size_t>(n) < sizeof(path))
			ready = (access(path, F_OK) == 0);
	}
	errno = saved_errno;
	return ready;
}

} // namespace agent_darwin
} // namespace bpftime
#endif // __APPLE__
