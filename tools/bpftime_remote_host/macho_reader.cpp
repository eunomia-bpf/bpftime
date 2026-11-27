/* SPDX-License-Identifier: MIT
 *
 * Mach-O / Fat binary / dyld_shared_cache symbol extraction.
 */
#include "macho_reader.hpp"

#include <fcntl.h>
#include <spdlog/spdlog.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <mach-o/fat.h>
#include <mach-o/loader.h>
#include <mach-o/nlist.h>
#include <set>
#include <sys/dtrace.h>

#if BPFTIME_HAVE_FRIDA_GUM
#include <frida-gum.h>
#endif

namespace bpftime
{
namespace remote
{

namespace
{

struct MappedFile {
	const uint8_t *data = nullptr;
	size_t size = 0;
	int fd = -1;

	~MappedFile()
	{
		if (data && data != MAP_FAILED)
			munmap((void *)data, size);
		if (fd >= 0)
			close(fd);
	}

	bool open(const std::string &path)
	{
		fd = ::open(path.c_str(), O_RDONLY);
		if (fd < 0)
			return false;
		struct stat st {};
		if (fstat(fd, &st) != 0 || st.st_size < (off_t)sizeof(uint32_t))
			return false;
		size = (size_t)st.st_size;
		data = (const uint8_t *)mmap(nullptr, size, PROT_READ,
					     MAP_PRIVATE, fd, 0);
		return data != nullptr && data != MAP_FAILED;
	}
};

// Mach-O C symbols carry one leading underscore; C++ mangled names keep
// their "_Z" prefix after removing it.
std::string normalize_symbol(const char *raw)
{
	std::string s(raw);
	if (!s.empty() && s[0] == '_')
		s.erase(s.begin());
	return s;
}

// Parse an Apple DTrace DOF blob (__TEXT,__dof_<provider> / __DATA,__dof)
// using the system <sys/dtrace.h> layout. On macOS arm64, ld64 emits each
// probe's nop callsite offset in DOF_SECT_PROFFS as a signed 32-bit offset
// relative to the DOF section's Mach-O vmaddr when dofpr_addr == 0, and
// probe arguments follow AAPCS64 in x0..x7 at the patched callsite.
void parse_dof_section(const uint8_t *dof, size_t dof_size,
		       uint64_t sect_vmaddr, uint64_t text_vmaddr,
		       std::vector<StapSdtEntry> &out)
{
	if (dof_size < sizeof(dof_hdr_t))
		return;
	const auto *hdr = reinterpret_cast<const dof_hdr_t *>(dof);
	if (hdr->dofh_ident[DOF_ID_MAG0] != DOF_MAG_MAG0 ||
	    hdr->dofh_ident[DOF_ID_MAG1] != DOF_MAG_MAG1 ||
	    hdr->dofh_ident[DOF_ID_MAG2] != DOF_MAG_MAG2 ||
	    hdr->dofh_ident[DOF_ID_MAG3] != DOF_MAG_MAG3)
		return;
	if (hdr->dofh_secsize < sizeof(dof_sec_t) ||
	    hdr->dofh_secoff +
			    uint64_t(hdr->dofh_secnum) * hdr->dofh_secsize >
		    dof_size)
		return;

	auto get_sec = [&](uint32_t idx) -> const dof_sec_t * {
		if (idx >= hdr->dofh_secnum)
			return nullptr;
		const auto *s = reinterpret_cast<const dof_sec_t *>(
			dof + hdr->dofh_secoff +
			uint64_t(idx) * hdr->dofh_secsize);
		if (s->dofs_offset + s->dofs_size > dof_size)
			return nullptr;
		return s;
	};

	for (uint32_t i = 0; i < hdr->dofh_secnum; i++) {
		const dof_sec_t *pv_sec = get_sec(i);
		if (pv_sec == nullptr || pv_sec->dofs_type != DOF_SECT_PROVIDER ||
		    pv_sec->dofs_size < sizeof(dof_provider_t))
			continue;
		const auto *pv = reinterpret_cast<const dof_provider_t *>(
			dof + pv_sec->dofs_offset);
		const dof_sec_t *str_sec = get_sec(pv->dofpv_strtab);
		const dof_sec_t *prb_sec = get_sec(pv->dofpv_probes);
		const dof_sec_t *off_sec = get_sec(pv->dofpv_proffs);
		if (str_sec == nullptr || prb_sec == nullptr ||
		    off_sec == nullptr || prb_sec->dofs_entsize < sizeof(dof_probe_t))
			continue;

		const char *strings =
			reinterpret_cast<const char *>(dof + str_sec->dofs_offset);
		auto safe_str = [&](uint32_t idx) -> std::string {
			if (idx >= str_sec->dofs_size)
				return {};
			size_t max_len = str_sec->dofs_size - idx;
			return std::string(strings + idx,
					   strnlen(strings + idx, max_len));
		};
		std::string provider_name = safe_str(pv->dofpv_name);
		if (provider_name.empty())
			continue;

		const auto *offs = reinterpret_cast<const uint32_t *>(
			dof + off_sec->dofs_offset);
		size_t total_offs = off_sec->dofs_size / sizeof(uint32_t);
		size_t nprobes = prb_sec->dofs_size / prb_sec->dofs_entsize;
		for (size_t pi = 0; pi < nprobes; pi++) {
			const auto *pr = reinterpret_cast<const dof_probe_t *>(
				dof + prb_sec->dofs_offset +
				pi * prb_sec->dofs_entsize);
			std::string probe_name = safe_str(pr->dofpr_name);
			if (probe_name.empty())
				continue;
			std::string argfmt;
			uint8_t nargc = std::min<uint8_t>(pr->dofpr_nargc, 8);
			for (uint8_t a = 0; a < nargc; a++) {
				if (a > 0)
					argfmt.push_back(' ');
				argfmt += "8@x" + std::to_string(a);
			}
			uint64_t base_vm =
				pr->dofpr_addr != 0 ? pr->dofpr_addr : sect_vmaddr;
			for (uint16_t k = 0; k < pr->dofpr_noffs; k++) {
				size_t oidx = size_t(pr->dofpr_offidx) + k;
				if (oidx >= total_offs)
					break;
				int64_t rel = static_cast<int32_t>(offs[oidx]);
				uint64_t site_vm =
					static_cast<uint64_t>(int64_t(base_vm) + rel);
				if (site_vm < text_vmaddr)
					continue;
				StapSdtEntry note;
				note.addr = site_vm - text_vmaddr;
				note.provider = provider_name;
				note.probe = probe_name;
				note.argfmt = argfmt;
				out.push_back(std::move(note));
			}
		}
	}
}

void parse_macho_64(const uint8_t *base, size_t size, ModuleSymbols &out)
{
	auto *mh = reinterpret_cast<const mach_header_64 *>(base);
	if (mh->magic != MH_MAGIC_64 ||
	    mh->cputype != CPU_TYPE_ARM64)
		return;
	const uint8_t *p = base + sizeof(mach_header_64);
	const symtab_command *symtab = nullptr;
	std::vector<std::pair<const section_64 *, uint64_t>> dof_sections;
	uint64_t text_vmaddr = 0;
	for (uint32_t i = 0; i < mh->ncmds; i++) {
		auto *cmd = reinterpret_cast<const load_command *>(p);
		if (p + cmd->cmdsize > base + size)
			break;
		if (cmd->cmd == LC_SEGMENT_64) {
			auto *seg =
				reinterpret_cast<const segment_command_64 *>(
					cmd);
			if (strcmp(seg->segname, "__TEXT") == 0) {
				text_vmaddr = seg->vmaddr;
				out.text_size = seg->vmsize;
			}
			auto *sp = reinterpret_cast<const uint8_t *>(seg) +
				   sizeof(segment_command_64);
			for (uint32_t s = 0; s < seg->nsects; s++) {
				auto *sect =
					reinterpret_cast<const section_64 *>(
						sp);
				// macOS ld names DOF sections __dof_<provider> in
				// __TEXT (the generic name is __dof); match the
				// family by prefix.
				if (strncmp(sect->sectname, "__dof", 5) == 0)
					dof_sections.emplace_back(sect,
								  (uint64_t)0);
				sp += sizeof(section_64);
			}
		} else if (cmd->cmd == LC_SYMTAB) {
			symtab =
				reinterpret_cast<const symtab_command *>(cmd);
		}
		p += cmd->cmdsize;
	}
	out.text_vmaddr = text_vmaddr;
	if (symtab != nullptr &&
	    base + symtab->symoff +
			    (size_t)symtab->nsyms * sizeof(nlist_64) <=
		    base + size &&
	    base + symtab->stroff + symtab->strsize <= base + size) {
		const char *strtab =
			reinterpret_cast<const char *>(base + symtab->stroff);
		auto *syms = reinterpret_cast<const nlist_64 *>(
			base + symtab->symoff);
		std::set<std::string> seen;
		for (uint32_t i = 0; i < symtab->nsyms; i++) {
			const nlist_64 &n = syms[i];
			if ((n.n_type & N_STAB) || (n.n_type & N_PEXT) ||
			    n.n_value == 0 || n.n_un.n_strx == 0)
				continue;
			if ((n.n_type & N_TYPE) != N_SECT)
				continue;
			if (n.n_value < text_vmaddr)
				continue;
			if (out.text_size > 0 &&
			    (n.n_value - text_vmaddr) >= out.text_size)
				continue;
			const char *raw = strtab + n.n_un.n_strx;
			if (raw >= strtab + symtab->strsize)
				continue;
			SymbolEntry e;
			e.name = normalize_symbol(raw);
			if (e.name.empty() || !seen.insert(e.name).second)
				continue;
			e.offset = n.n_value - text_vmaddr;
			out.symbols.push_back(std::move(e));
		}
	}
	for (auto [sect, _] : dof_sections) {
		if ((size_t)sect->offset + sect->size > size)
			continue;
		parse_dof_section(base + sect->offset, sect->size, sect->addr,
				  text_vmaddr, out.sdt_notes);
	}
}

} // namespace

bool is_dyld_virtual_path(const std::string &path)
{
	return path.rfind("/bpftime-dyld/", 0) == 0;
}

std::string dyld_virtual_to_real(const std::string &path)
{
	if (is_dyld_virtual_path(path))
		return path.substr(strlen("/bpftime-dyld"));
	return path;
}

std::string dyld_real_to_virtual(const std::string &path)
{
	if (path.rfind("/usr/lib/", 0) == 0 ||
	    path.rfind("/System/Library/", 0) == 0)
		return "/bpftime-dyld" + path;
	return path;
}

ModuleSymbols parse_disk_macho(const std::string &path)
{
	ModuleSymbols out;
	out.canonical_path = path;
	MappedFile file;
	if (!file.open(path)) {
		SPDLOG_WARN("synth_elf: cannot open {}", path);
		return out;
	}
	uint32_t magic;
	memcpy(&magic, file.data, sizeof(magic));
	if (magic == MH_MAGIC_64) {
		parse_macho_64(file.data, file.size, out);
	} else if (magic == FAT_CIGAM || magic == FAT_MAGIC) {
		auto *fh =
			reinterpret_cast<const fat_header *>(file.data);
		uint32_t n = ntohl(fh->nfat_arch);
		bool be = (magic == FAT_CIGAM);
		const auto *archs = reinterpret_cast<const fat_arch *>(
			file.data + sizeof(fat_header));
		for (uint32_t i = 0; i < n; i++) {
			uint32_t cputype = be ? ntohl(archs[i].cputype) :
						archs[i].cputype;
			uint32_t offset = be ? ntohl(archs[i].offset) :
					       archs[i].offset;
			uint32_t asize = be ? ntohl(archs[i].size) :
					      archs[i].size;
			if (cputype != CPU_TYPE_ARM64)
				continue;
			if ((size_t)offset + asize > file.size)
				break;
			parse_macho_64(file.data + offset, asize, out);
			break;
		}
	} else {
		SPDLOG_WARN("synth_elf: unrecognized Mach-O magic 0x{:x} in {}",
			    magic, path);
	}
	std::sort(out.symbols.begin(), out.symbols.end(),
		  [](const SymbolEntry &a, const SymbolEntry &b) {
			  return a.name < b.name;
		  });
	return out;
}

#if BPFTIME_HAVE_FRIDA_GUM
namespace
{
inline uintptr_t strip_pac_ptr(uintptr_t p)
{
	return p & 0x0000007fffffffffULL;
}

uint64_t find_loaded_text_size(const std::string &real_path)
{
	for (uint32_t i = 0; i < _dyld_image_count(); i++) {
		const char *img = _dyld_get_image_name(i);
		if (img == nullptr || real_path != img)
			continue;
		const auto *mh = reinterpret_cast<const mach_header_64 *>(
			_dyld_get_image_header(i));
		if (mh == nullptr || mh->magic != MH_MAGIC_64)
			break;
		const auto *cmd = reinterpret_cast<const load_command *>(
			reinterpret_cast<const uint8_t *>(mh) +
			sizeof(mach_header_64));
		for (uint32_t c = 0; c < mh->ncmds;
		     c++, cmd = reinterpret_cast<const load_command *>(
				  reinterpret_cast<const uint8_t *>(cmd) +
				  cmd->cmdsize)) {
			if (cmd->cmd == LC_SEGMENT_64) {
				const auto *seg = reinterpret_cast<
					const segment_command_64 *>(cmd);
				if (strcmp(seg->segname, "__TEXT") == 0)
					return seg->vmsize;
			}
		}
		break;
	}
	return 0;
}

struct ExportCollector {
	std::vector<SymbolEntry> symbols;
	uintptr_t base;
	uint64_t max_span;
};

gboolean export_cb(const GumExportDetails *details, gpointer user_data)
{
	auto *ctx = static_cast<ExportCollector *>(user_data);
	if (details->type != GUM_EXPORT_FUNCTION)
		return TRUE;
	uintptr_t addr = strip_pac_ptr((uintptr_t)details->address);
	if (addr < ctx->base)
		return TRUE;
	uint64_t rel = (uint64_t)(addr - ctx->base);
	if (rel >= ctx->max_span)
		return TRUE;
	SymbolEntry e;
	e.name = normalize_symbol(details->name);
	if (e.name.empty())
		return TRUE;
	e.offset = rel;
	ctx->symbols.push_back(std::move(e));
	return TRUE;
}
} // namespace
#endif

ModuleSymbols enumerate_runtime_exports(const std::string &real_path)
{
	ModuleSymbols out;
	out.canonical_path = real_path;
#if BPFTIME_HAVE_FRIDA_GUM
	void *handle = dlopen(real_path.c_str(), RTLD_LAZY | RTLD_NOLOAD);
	if (handle == nullptr)
		handle = dlopen(real_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
	if (handle == nullptr) {
		SPDLOG_WARN("synth_elf: dlopen failed for {}: {}", real_path,
			    dlerror());
		return out;
	}
	out.text_size = find_loaded_text_size(real_path);
	uintptr_t base = strip_pac_ptr(
		(uintptr_t)gum_module_find_base_address(real_path.c_str()));
	if (base == 0) {
		// Frida matches the basename for some shared cache images.
		const char *slash = strrchr(real_path.c_str(), '/');
		base = strip_pac_ptr((uintptr_t)gum_module_find_base_address(
			slash ? slash + 1 : real_path.c_str()));
	}
	if (base == 0) {
		SPDLOG_WARN("synth_elf: no base address for {}", real_path);
		return out;
	}
	const uint64_t max_span =
		out.text_size > 0 ? out.text_size : (64ULL << 20);
	ExportCollector collector{ {}, base, max_span };
	gum_module_enumerate_exports(real_path.c_str(), export_cb, &collector);
	if (collector.symbols.empty()) {
		const char *slash = strrchr(real_path.c_str(), '/');
		if (slash)
			gum_module_enumerate_exports(slash + 1, export_cb,
						     &collector);
	}
	if (real_path.rfind("/usr/lib/system/", 0) == 0) {
		ModuleSymbols local = enumerate_runtime_symbols(real_path);
		for (auto &ls : local.symbols) {
			ls.name = normalize_symbol(ls.name.c_str());
			if (!ls.name.empty() && ls.offset < max_span)
				collector.symbols.push_back(std::move(ls));
		}
	}
	std::sort(collector.symbols.begin(), collector.symbols.end(),
		  [](const SymbolEntry &a, const SymbolEntry &b) {
			  return a.name < b.name;
		  });
	collector.symbols.erase(
		std::unique(collector.symbols.begin(),
			    collector.symbols.end(),
			    [](const SymbolEntry &a, const SymbolEntry &b) {
				    return a.name == b.name;
			    }),
		collector.symbols.end());
	out.symbols = std::move(collector.symbols);
#else
	SPDLOG_ERROR("synth_elf: built without Frida Gum; cannot enumerate {}",
		     real_path);
#endif
	return out;
}

namespace
{
struct SymbolCollector {
	std::vector<SymbolEntry> entries;
	std::set<std::string> seen;
};

static gboolean symbol_collect_cb(const GumSymbolDetails *details,
				  gpointer user_data)
{
	auto *ctx = static_cast<SymbolCollector *>(user_data);
	if (!details->name || details->address == 0)
		return TRUE;
	std::string name(details->name);
	if (name.empty() || !ctx->seen.insert(name).second)
		return TRUE;
	ctx->entries.push_back(SymbolEntry{
		name, (uint64_t)strip_pac_ptr((uintptr_t)details->address) });
	return TRUE;
}
} // namespace

ModuleSymbols enumerate_runtime_symbols(const std::string &real_path)
{
	ModuleSymbols out;
	out.canonical_path = real_path;
#if BPFTIME_HAVE_FRIDA_GUM
	void *handle = dlopen(real_path.c_str(), RTLD_LAZY | RTLD_NOLOAD);
	if (handle == nullptr)
		handle = dlopen(real_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
	if (handle == nullptr)
		return out;
	out.text_size = find_loaded_text_size(real_path);
	uintptr_t base = strip_pac_ptr(
		(uintptr_t)gum_module_find_base_address(real_path.c_str()));
	if (base == 0)
		return out;
	const uint64_t max_span =
		out.text_size > 0 ? out.text_size : (64ULL << 20);
	SymbolCollector collector;
	gum_module_enumerate_symbols(real_path.c_str(), symbol_collect_cb,
				     &collector);
	out.symbols.reserve(collector.entries.size());
	for (auto &e : collector.entries) {
		if (e.offset < base)
			continue;
		uint64_t rel = e.offset - base;
		if (rel >= max_span)
			continue;
		e.offset = rel;
		out.symbols.push_back(std::move(e));
	}
#endif
	return out;
}

DeepSymbol resolve_deep_symbol(const std::string &real_module,
			       const std::vector<std::string> &candidates)
{
	DeepSymbol result;
	ModuleSymbols syms = enumerate_runtime_symbols(real_module);
	for (const auto &wanted : candidates) {
		auto it = std::find_if(syms.symbols.begin(), syms.symbols.end(),
				       [&](const SymbolEntry &e) {
					       return e.name == wanted;
				       });
		if (it != syms.symbols.end()) {
			result.module = real_module;
			result.offset = it->offset;
			return result;
		}
	}
	return result;
}
UmbrellaIndex build_libsystem_umbrella(const std::string &virtual_path)
{
	static const char *kChildren[] = {
		"/usr/lib/system/libsystem_malloc.dylib",
		"/usr/lib/system/libsystem_c.dylib",
		"/usr/lib/system/libsystem_kernel.dylib",
		"/usr/lib/system/libsystem_pthread.dylib",
		"/usr/lib/system/libsystem_platform.dylib",
	};
	UmbrellaIndex idx;
	idx.stub.canonical_path = virtual_path;
#if BPFTIME_HAVE_FRIDA_GUM
	idx.stub.text_size =
		find_loaded_text_size("/usr/lib/libSystem.B.dylib");
#endif
	std::set<std::string> seen;
	uint64_t synth_off = UMBRELLA_SYNTH_BASE;
	for (const char *child : kChildren) {
		ModuleSymbols child_syms = enumerate_runtime_exports(child);
		for (const auto &sym : child_syms.symbols) {
			if (!seen.insert(sym.name).second)
				continue;
			idx.translations[synth_off] = { child, sym.offset };
			idx.stub.symbols.push_back(
				SymbolEntry{ sym.name, synth_off });
			synth_off += UMBRELLA_SYNTH_STRIDE;
		}
	}
	// The libmalloc "malloc" export is a PAC re-export veneer that lazy
	// binding bypasses after the first call, so Frida hooks on it do not
	// fire under DYLD_INSERT. Resolve the deep allocator entry that real
	// calls use and record it for perf_event_open translation.
	static const char *kMallocImpls[] = {
		"_xzm_xzone_malloc", "malloc_type_zone_malloc",
		"szone_malloc", "nanov2_malloc", "malloc",
	};
	DeepSymbol malloc_deep = resolve_deep_symbol(
		"/usr/lib/system/libsystem_malloc.dylib",
		std::vector<std::string>(kMallocImpls,
					 kMallocImpls +
						 sizeof(kMallocImpls) /
							 sizeof(kMallocImpls[0])));
	if (!malloc_deep.module.empty())
		idx.deep_symbols["malloc"] = std::move(malloc_deep);
	static const char *kFreeImpls[] = {
		"_xzm_free", "szone_free", "nanov2_free", "free",
	};
	DeepSymbol free_deep = resolve_deep_symbol(
		"/usr/lib/system/libsystem_malloc.dylib",
		std::vector<std::string>(kFreeImpls,
					 kFreeImpls +
						 sizeof(kFreeImpls) /
							 sizeof(kFreeImpls[0])));
	if (!free_deep.module.empty())
		idx.deep_symbols["free"] = std::move(free_deep);
	std::sort(idx.stub.symbols.begin(), idx.stub.symbols.end(),
		  [](const SymbolEntry &a, const SymbolEntry &b) {
			  return a.name < b.name;
		  });
	return idx;
}

} // namespace remote
} // namespace bpftime
