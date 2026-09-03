/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "trap_attach_utils.hpp"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <link.h>
#include <spdlog/spdlog.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#include <vector>

namespace bpftime::attach::trap
{
namespace
{
std::string get_executable_path()
{
	char exec_path[PATH_MAX] = { 0 };
	ssize_t len =
		readlink("/proc/self/exe", exec_path, sizeof(exec_path) - 1);
	if (len < 0) {
		SPDLOG_ERROR("Error retrieving executable path: {}", errno);
		return "";
	}
	exec_path[len] = '\0';
	return exec_path;
}

bool same_file(const std::string &a, const std::string &b)
{
	if (a == b)
		return true;
	std::error_code ec;
	bool eq = std::filesystem::equivalent(a, b, ec);
	return !ec && eq;
}

std::string basename_of(const std::string &path)
{
	auto pos = path.find_last_of('/');
	return pos == std::string::npos ? path : path.substr(pos + 1);
}

struct loaded_module {
	std::string path;
	// Difference between link-time and runtime addresses
	uintptr_t load_bias;
	// Runtime address of file offset 0
	uintptr_t base;
};

std::vector<loaded_module> enumerate_modules()
{
	std::vector<loaded_module> result;
	auto exe = get_executable_path();
	dl_iterate_phdr(
		[](struct dl_phdr_info *info, size_t, void *data) -> int {
			auto &out = *(std::vector<loaded_module> *)data;
			loaded_module m;
			m.path = info->dlpi_name ? info->dlpi_name : "";
			m.load_bias = info->dlpi_addr;
			uintptr_t base = 0;
			bool found = false;
			for (int i = 0; i < info->dlpi_phnum; i++) {
				const auto &ph = info->dlpi_phdr[i];
				if (ph.p_type != PT_LOAD)
					continue;
				uintptr_t start = info->dlpi_addr + ph.p_vaddr -
						  ph.p_offset;
				if (!found || start < base) {
					base = start;
					found = true;
				}
			}
			m.base = found ? base : info->dlpi_addr;
			out.push_back(std::move(m));
			return 0;
		},
		&result);
	// The first entry is the main executable and has an empty name
	if (!result.empty() && result[0].path.empty())
		result[0].path = exe;
	return result;
}

bool module_matches(const loaded_module &m, const std::string &wanted)
{
	if (wanted.empty())
		return false;
	if (same_file(m.path, wanted))
		return true;
	// Allow matching by file name (e.g. "libc.so.6")
	return wanted.find('/') == std::string::npos &&
	       basename_of(m.path) == wanted;
}

// Scan the symbol tables of the ELF file backing `m` for `name`.
void *lookup_symbol_in_file(const loaded_module &m, const char *name)
{
	int fd = open(m.path.c_str(), O_RDONLY | O_CLOEXEC);
	if (fd < 0)
		return nullptr;
	struct stat st;
	if (fstat(fd, &st) < 0 || st.st_size < (off_t)sizeof(Elf64_Ehdr)) {
		close(fd);
		return nullptr;
	}
	size_t size = (size_t)st.st_size;
	void *map = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
	close(fd);
	if (map == MAP_FAILED)
		return nullptr;
	void *result = nullptr;
	const auto *bytes = (const uint8_t *)map;
	const auto *ehdr = (const Elf64_Ehdr *)bytes;
	if (std::memcmp(ehdr->e_ident, ELFMAG, SELFMAG) == 0 &&
	    ehdr->e_ident[EI_CLASS] == ELFCLASS64 && ehdr->e_shoff != 0 &&
	    ehdr->e_shentsize == sizeof(Elf64_Shdr) &&
	    ehdr->e_shoff + (size_t)ehdr->e_shnum * sizeof(Elf64_Shdr) <=
		    size) {
		const auto *shdrs = (const Elf64_Shdr *)(bytes + ehdr->e_shoff);
		for (unsigned i = 0; i < ehdr->e_shnum && !result; i++) {
			const auto &sh = shdrs[i];
			if (sh.sh_type != SHT_SYMTAB && sh.sh_type != SHT_DYNSYM)
				continue;
			if (sh.sh_link >= ehdr->e_shnum ||
			    sh.sh_entsize != sizeof(Elf64_Sym) ||
			    sh.sh_offset + sh.sh_size > size)
				continue;
			const auto &strsh = shdrs[sh.sh_link];
			if (strsh.sh_offset + strsh.sh_size > size)
				continue;
			const char *strtab = (const char *)(bytes + strsh.sh_offset);
			const auto *syms = (const Elf64_Sym *)(bytes + sh.sh_offset);
			size_t count = sh.sh_size / sizeof(Elf64_Sym);
			for (size_t j = 0; j < count; j++) {
				const auto &sym = syms[j];
				unsigned type = ELF64_ST_TYPE(sym.st_info);
				if (sym.st_shndx == SHN_UNDEF || sym.st_value == 0)
					continue;
				if (type != STT_FUNC && type != STT_NOTYPE)
					continue;
				if (sym.st_name >= strsh.sh_size)
					continue;
				if (std::strcmp(strtab + sym.st_name, name) != 0)
					continue;
				result = (void *)(m.load_bias + sym.st_value);
				break;
			}
		}
	}
	munmap(map, size);
	return result;
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

void *get_module_base_addr(const char *module_name)
{
	std::string wanted = module_name ? module_name : "";
	auto modules = enumerate_modules();
	if (modules.empty())
		return nullptr;
	if (wanted.empty() || same_file(wanted, modules[0].path))
		return (void *)modules[0].base;
	for (const auto &m : modules) {
		if (module_matches(m, wanted))
			return (void *)m.base;
	}
	return nullptr;
}

void *
resolve_function_addr_by_module_offset(const std::string_view &module_name,
				       uintptr_t func_offset)
{
	void *base = get_module_base_addr(std::string(module_name).c_str());
	if (!base) {
		// Not necessarily a bug: with LD_PRELOAD the agent might have
		// been loaded into an unrelated process
		SPDLOG_INFO("Failed to find module base address for {}",
			    module_name);
		return nullptr;
	}
	return (char *)base + func_offset;
}

std::optional<std::string>
resolve_mapped_module_path(const std::string_view &module_name)
{
	std::string name(module_name);
	unsigned pid = 0;
	unsigned long long wanted_start = 0, wanted_end = 0;
	int consumed = 0;
	if (sscanf(name.c_str(), "/proc/%u/map_files/%llx-%llx%n", &pid,
		   &wanted_start, &wanted_end, &consumed) != 3 ||
	    consumed != static_cast<int>(name.size()))
		return name;
	if ((pid_t)pid != getpid() || wanted_start >= wanted_end)
		return {};

	std::ifstream maps("/proc/self/maps");
	std::string line;
	while (std::getline(maps, line)) {
		unsigned long long start, end, offset, inode;
		unsigned dev_major, dev_minor;
		char permissions[5] = {};
		int path_offset = 0;
		if (sscanf(line.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n",
			   &start, &end, permissions, &offset, &dev_major,
			   &dev_minor, &inode, &path_offset) != 7 ||
		    end <= wanted_start || start >= wanted_end)
			continue;
		std::string path = unescape_proc_path(line.substr(path_offset));
		struct stat st = {};
		if (path.empty() || path.front() != '/' ||
		    path.ends_with(" (deleted)") ||
		    stat(path.c_str(), &st) != 0 ||
		    (inode != 0 && st.st_ino != inode) ||
		    major(st.st_dev) != dev_major ||
		    minor(st.st_dev) != dev_minor)
			return {};
		return path;
	}
	return {};
}

void *find_function_addr_by_name(const char *name)
{
	if (!name)
		return nullptr;
	if (void *p = dlsym(RTLD_DEFAULT, name); p)
		return p;
	for (const auto &m : enumerate_modules()) {
		if (m.path.empty())
			continue;
		if (void *p = lookup_symbol_in_file(m, name); p)
			return p;
	}
	return nullptr;
}

void *find_module_export_by_name(const char *module_name,
				 const char *symbol_name)
{
	if (!symbol_name)
		return nullptr;
	if (!module_name || module_name[0] == '\0')
		return dlsym(RTLD_DEFAULT, symbol_name);
	void *handle = dlopen(module_name, RTLD_LAZY | RTLD_NOLOAD);
	if (!handle)
		return nullptr;
	void *result = dlsym(handle, symbol_name);
	dlclose(handle);
	return result;
}
} // namespace bpftime::attach::trap
