#include "nv_elf_introspect.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <elf.h>
#include <fstream>
#include <link.h>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>
#include <unistd.h>

namespace bpftime::attach::elf_introspect {

namespace {

std::string read_self_exe_path()
{
#if defined(__linux__)
	char buf[4096];
	ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
	if (n <= 0)
		return {};
	buf[n] = '\0';
	return std::string(buf);
#else
	return {};
#endif
}

bool read_exact(std::ifstream &ifs, std::uint64_t off, void *out,
		std::size_t len)
{
	ifs.seekg((std::streamoff)off, std::ios::beg);
	if (!ifs.good())
		return false;
	ifs.read(reinterpret_cast<char *>(out), (std::streamsize)len);
	return ifs.good();
}

struct section_span {
	std::uint64_t sh_addr = 0;
	std::uint64_t sh_offset = 0;
	std::uint64_t sh_size = 0;
	std::uint32_t sh_type = 0;
	std::uint32_t sh_link = 0;
};

template <typename Ehdr, typename Shdr, typename Sym>
std::optional<section_span>
find_section_span(std::ifstream &ifs, std::string_view section_name)
{
	Ehdr eh;
	if (!read_exact(ifs, 0, &eh, sizeof(eh)))
		return std::nullopt;
	if (eh.e_shoff == 0 || eh.e_shnum == 0)
		return std::nullopt;

	// Load section header string table header.
	Shdr shstr;
	const std::uint64_t shstr_off =
		(std::uint64_t)eh.e_shoff +
		(std::uint64_t)eh.e_shentsize * (std::uint64_t)eh.e_shstrndx;
	if (!read_exact(ifs, shstr_off, &shstr, sizeof(shstr)))
		return std::nullopt;

	std::string shstrtab;
	shstrtab.resize((std::size_t)shstr.sh_size);
	if (!read_exact(ifs, (std::uint64_t)shstr.sh_offset, shstrtab.data(),
			shstrtab.size()))
		return std::nullopt;

	for (std::uint16_t i = 0; i < eh.e_shnum; i++) {
		Shdr sh;
		const std::uint64_t off = (std::uint64_t)eh.e_shoff +
					  (std::uint64_t)eh.e_shentsize *
						  (std::uint64_t)i;
		if (!read_exact(ifs, off, &sh, sizeof(sh)))
			return std::nullopt;

		const char *nm = "";
		if (sh.sh_name < shstrtab.size())
			nm = shstrtab.data() + sh.sh_name;
		if (section_name == nm) {
			return section_span{
				.sh_addr = (std::uint64_t)sh.sh_addr,
				.sh_offset = (std::uint64_t)sh.sh_offset,
				.sh_size = (std::uint64_t)sh.sh_size,
				.sh_type = (std::uint32_t)sh.sh_type,
				.sh_link = (std::uint32_t)sh.sh_link,
			};
		}
	}
	return std::nullopt;
}

template <typename Ehdr, typename Shdr, typename Sym>
std::vector<symbol_range>
read_function_symbols_impl(std::ifstream &ifs, std::uintptr_t base)
{
	std::vector<symbol_range> out;

	Ehdr eh;
	if (!read_exact(ifs, 0, &eh, sizeof(eh)))
		return out;
	if (eh.e_shoff == 0 || eh.e_shnum == 0)
		return out;

	std::vector<Shdr> shdrs;
	shdrs.resize(eh.e_shnum);
	for (std::uint16_t i = 0; i < eh.e_shnum; i++) {
		const std::uint64_t off = (std::uint64_t)eh.e_shoff +
					  (std::uint64_t)eh.e_shentsize *
						  (std::uint64_t)i;
		if (!read_exact(ifs, off, &shdrs[i], sizeof(Shdr)))
			return {};
	}

	for (std::uint16_t i = 0; i < eh.e_shnum; i++) {
		const auto &sh = shdrs[i];
		if (sh.sh_type != SHT_SYMTAB && sh.sh_type != SHT_DYNSYM)
			continue;
		if (sh.sh_entsize == 0)
			continue;
		if (sh.sh_link >= shdrs.size())
			continue;
		const auto &str_sh = shdrs[sh.sh_link];
		if (str_sh.sh_size == 0)
			continue;
		std::string strtab;
		strtab.resize((std::size_t)str_sh.sh_size);
		if (!read_exact(ifs, (std::uint64_t)str_sh.sh_offset,
				strtab.data(), strtab.size()))
			continue;

		const std::size_t sym_count =
			(std::size_t)(sh.sh_size / sh.sh_entsize);
		for (std::size_t si = 0; si < sym_count; si++) {
			Sym sym;
			const std::uint64_t sym_off =
				(std::uint64_t)sh.sh_offset +
				(std::uint64_t)sh.sh_entsize *
					(std::uint64_t)si;
			if (!read_exact(ifs, sym_off, &sym, sizeof(sym)))
				break;

			const unsigned char type = ELF64_ST_TYPE(sym.st_info);
			if (type != STT_FUNC)
				continue;
			if (sym.st_value == 0)
				continue;
			const char *name = "";
			if (sym.st_name < strtab.size())
				name = strtab.data() + sym.st_name;
			if (name[0] == '\0')
				continue;
			std::uintptr_t start = base + (std::uintptr_t)sym.st_value;
			std::uintptr_t end = start;
			if (sym.st_size)
				end = start + (std::uintptr_t)sym.st_size;
			out.push_back(symbol_range{
				.start = start,
				.end = end,
				.name = std::string(name),
			});
		}
	}

	std::sort(out.begin(), out.end(),
		  [](const symbol_range &a, const symbol_range &b) {
			  return a.start < b.start;
		  });
	return out;
}

std::string normalize_module_path(std::string path)
{
	if (!path.empty())
		return path;
	return read_self_exe_path();
}

} // namespace

std::vector<loaded_module> list_loaded_modules()
{
	std::vector<loaded_module> out;
	dl_iterate_phdr(
		[](struct dl_phdr_info *info, size_t, void *data) -> int {
			auto &vec = *static_cast<std::vector<loaded_module> *>(
				data);
			loaded_module mod;
			if (info->dlpi_name != nullptr)
				mod.path = normalize_module_path(info->dlpi_name);
			mod.base_addr = (std::uintptr_t)info->dlpi_addr;
			if (!mod.path.empty())
				vec.push_back(std::move(mod));
			return 0;
		},
		&out);
	return out;
}

std::optional<std::pair<const void *, std::size_t>>
find_section_in_memory(const loaded_module &mod, std::string_view section_name)
{
	const std::string path = normalize_module_path(mod.path);
	if (path.empty())
		return std::nullopt;
	std::ifstream ifs(path, std::ios::binary);
	if (!ifs.is_open())
		return std::nullopt;

	unsigned char ident[EI_NIDENT];
	if (!read_exact(ifs, 0, ident, sizeof(ident)))
		return std::nullopt;
	if (memcmp(ident, ELFMAG, SELFMAG) != 0)
		return std::nullopt;

	std::optional<section_span> sec;
	if (ident[EI_CLASS] == ELFCLASS64) {
		sec = find_section_span<Elf64_Ehdr, Elf64_Shdr, Elf64_Sym>(
			ifs, section_name);
	} else if (ident[EI_CLASS] == ELFCLASS32) {
		sec = find_section_span<Elf32_Ehdr, Elf32_Shdr, Elf32_Sym>(
			ifs, section_name);
	} else {
		return std::nullopt;
	}
	if (!sec || sec->sh_addr == 0 || sec->sh_size == 0)
		return std::nullopt;

	const auto *addr =
		reinterpret_cast<const void *>(mod.base_addr + sec->sh_addr);
	return std::make_pair(addr, (std::size_t)sec->sh_size);
}

std::vector<const __fatBinC_Wrapper_t *>
scan_fatbin_wrappers(const void *section_addr, std::size_t section_size)
{
	std::vector<const __fatBinC_Wrapper_t *> out;
	if (section_addr == nullptr || section_size < sizeof(__fatBinC_Wrapper_t))
		return out;

	// Magic used by CUDA fatbin wrapper: 'FbC1'
	constexpr std::uint32_t kWrapperMagic = 0x466243b1u;
	const auto *base = reinterpret_cast<const std::uint8_t *>(section_addr);
	for (std::size_t off = 0; off + sizeof(__fatBinC_Wrapper_t) <= section_size;
	     off += 8) {
		auto *w = reinterpret_cast<const __fatBinC_Wrapper_t *>(base + off);
		if ((std::uint32_t)w->magic != kWrapperMagic)
			continue;
		if (w->data == nullptr)
			continue;
		out.push_back(w);
	}
	return out;
}

std::vector<symbol_range> read_function_symbols(const loaded_module &mod)
{
	std::vector<symbol_range> out;
	const std::string path = normalize_module_path(mod.path);
	if (path.empty())
		return out;
	std::ifstream ifs(path, std::ios::binary);
	if (!ifs.is_open())
		return out;

	unsigned char ident[EI_NIDENT];
	if (!read_exact(ifs, 0, ident, sizeof(ident)))
		return out;
	if (memcmp(ident, ELFMAG, SELFMAG) != 0)
		return out;

	if (ident[EI_CLASS] == ELFCLASS64) {
		return read_function_symbols_impl<Elf64_Ehdr, Elf64_Shdr, Elf64_Sym>(
			ifs, mod.base_addr);
	}
	if (ident[EI_CLASS] == ELFCLASS32) {
		return read_function_symbols_impl<Elf32_Ehdr, Elf32_Shdr, Elf32_Sym>(
			ifs, mod.base_addr);
	}
	return out;
}

} // namespace bpftime::attach::elf_introspect
