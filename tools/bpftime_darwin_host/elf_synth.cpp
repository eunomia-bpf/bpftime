/* SPDX-License-Identifier: MIT
 *
 * Minimal, dependency-free EM_AARCH64 ET_DYN ELF synthesizer.
 */
#include "elf_synth.hpp"
#include <algorithm>
#include <cstring>
#include <unordered_map>

namespace bpftime
{
namespace darwin
{

namespace
{

// ELF constants duplicated so the host builds without the Linux-only <elf.h>.
constexpr uint16_t EM_AARCH64 = 183;
constexpr uint16_t ET_DYN = 3;
constexpr uint32_t PT_LOAD_V = 1;
constexpr uint32_t PT_NOTE_V = 4;
constexpr uint32_t PF_X = 1;
constexpr uint32_t PF_R = 4;
constexpr uint32_t SHT_PROGBITS = 1;
constexpr uint32_t SHT_STRTAB = 3;
constexpr uint32_t SHT_DYNSYM = 11;
constexpr uint32_t SHT_SYMTAB = 2;
constexpr uint32_t SHF_ALLOC = 2;
constexpr uint32_t SHF_EXECINSTR = 4;
constexpr uint8_t STB_GLOBAL = 1;
constexpr uint8_t STT_FUNC = 2;
constexpr uint32_t NT_STAPSDT = 3;

struct Elf64_Ehdr_D {
	uint8_t e_ident[16];
	uint16_t e_type;
	uint16_t e_machine;
	uint32_t e_version;
	uint64_t e_entry;
	uint64_t e_phoff;
	uint64_t e_shoff;
	uint32_t e_flags;
	uint16_t e_ehsize;
	uint16_t e_phentsize;
	uint16_t e_phnum;
	uint16_t e_shentsize;
	uint16_t e_shnum;
	uint16_t e_shstrndx;
};

struct Elf64_Phdr_D {
	uint32_t p_type;
	uint32_t p_flags;
	uint64_t p_offset;
	uint64_t p_vaddr;
	uint64_t p_paddr;
	uint64_t p_filesz;
	uint64_t p_memsz;
	uint64_t p_align;
};

struct Elf64_Shdr_D {
	uint32_t sh_name;
	uint32_t sh_type;
	uint64_t sh_flags;
	uint64_t sh_addr;
	uint64_t sh_offset;
	uint64_t sh_size;
	uint32_t sh_link;
	uint32_t sh_info;
	uint64_t sh_addralign;
	uint64_t sh_entsize;
};

struct Elf64_Sym_D {
	uint32_t st_name;
	uint8_t st_info;
	uint8_t st_other;
	uint16_t st_shndx;
	uint64_t st_value;
	uint64_t st_size;
};

class ElfBuilder {
    public:
	std::vector<uint8_t> buf;

	void pad(size_t alignment)
	{
		size_t rem = buf.size() % alignment;
		if (rem)
			buf.resize(buf.size() + (alignment - rem), 0);
	}

	void append(const void *p, size_t n)
	{
		const auto *b = static_cast<const uint8_t *>(p);
		buf.insert(buf.end(), b, b + n);
	}

	template <typename T> void append_pod(const T &v)
	{
		append(&v, sizeof(T));
	}

	void patch_u16(size_t off, uint16_t v)
	{
		memcpy(buf.data() + off, &v, sizeof(v));
	}

	void patch_u64(size_t off, uint64_t v)
	{
		memcpy(buf.data() + off, &v, sizeof(v));
	}
};

struct StrTab {
	std::vector<uint8_t> bytes{ 0 }; // index 0 is the empty string
	std::unordered_map<std::string, uint32_t> offsets;

	uint32_t intern(const std::string &s)
	{
		auto it = offsets.find(s);
		if (it != offsets.end())
			return it->second;
		uint32_t off = (uint32_t)bytes.size();
		offsets[s] = off;
		bytes.insert(bytes.end(), s.begin(), s.end());
		bytes.push_back(0);
		return off;
	}
};

void append_le32(std::vector<uint8_t> &out, uint32_t v)
{
	out.push_back((uint8_t)(v & 0xff));
	out.push_back((uint8_t)((v >> 8) & 0xff));
	out.push_back((uint8_t)((v >> 16) & 0xff));
	out.push_back((uint8_t)((v >> 24) & 0xff));
}

void append_le64(std::vector<uint8_t> &out, uint64_t v)
{
	for (int i = 0; i < 8; i++)
		out.push_back((uint8_t)((v >> (8 * i)) & 0xff));
}

std::vector<uint8_t> build_stapsdt_notes(const std::vector<StapSdtEntry> &notes)
{
	std::vector<uint8_t> out;
	auto pad4 = [&]() {
		while (out.size() % 4)
			out.push_back(0);
	};
	for (const auto &n : notes) {
		const char owner[] = "stapsdt";
		uint32_t namesz = sizeof(owner); // includes NUL
		std::vector<uint8_t> desc;
		auto d64 = [&](uint64_t v) { append_le64(desc, v); };
		d64(n.addr); // probe address (module relative)
		d64(0); // base address
		d64(0); // semaphore
		desc.insert(desc.end(), n.provider.begin(), n.provider.end());
		desc.push_back(0);
		desc.insert(desc.end(), n.probe.begin(), n.probe.end());
		desc.push_back(0);
		desc.insert(desc.end(), n.argfmt.begin(), n.argfmt.end());
		desc.push_back(0);
		append_le32(out, namesz);
		append_le32(out, (uint32_t)desc.size());
		append_le32(out, NT_STAPSDT);
		out.insert(out.end(), owner, owner + namesz);
		pad4();
		out.insert(out.end(), desc.begin(), desc.end());
		pad4();
	}
	return out;
}

// Synthesize an ARM64 .eh_frame section (1 CIE + FDEs over [4, text_span))
// for DWARF-based eBPF stack unwinders. Function entries unwind LR (x30)
// with CFA=SP+0; bodies use the standard frame-pointer chain
// (CFA=x29+16, saved x29 at CFA-16, x30 at CFA-8). FDE starts are > 0
// because DWARF consumers drop FDEs at address 0.
std::vector<uint8_t>
build_arm64_eh_frame(const std::vector<uint64_t> &sorted_offsets,
		     uint64_t text_span)
{
	std::vector<uint8_t> out;

	// CIE at offset 0: 4-byte length(20) + 20-byte body.
	append_le32(out, 20);
	append_le32(out, 0); // CIE_id = 0 in .eh_frame
	out.push_back(1); // version
	out.push_back(0); // augmentation = ""
	out.push_back(4); // code_alignment_factor
	out.push_back(0x78); // data_alignment_factor = -8 (SLEB128)
	out.push_back(30); // return_address_register = x30
	// DW_CFA_def_cfa x31(SP), 0, then 8 NOPs.
	out.push_back(0x0c);
	out.push_back(31);
	out.push_back(0);
	for (int i = 0; i < 8; i++)
		out.push_back(0x00);

	auto emit_fde = [&](uint64_t start_addr, uint64_t end_addr,
			    bool is_symbol_start) {
		if (start_addr == 0)
			start_addr = 4;
		if (end_addr <= start_addr)
			return;
		uint64_t range = end_addr - start_addr;
		uint32_t fde_pos = (uint32_t)out.size();
		append_le32(out, 28); // 32-byte FDE = length + 28-byte body
		append_le32(out, fde_pos + 4); // CIE pointer -> CIE at 0
		append_le64(out, start_addr);
		append_le64(out, range);
		if (is_symbol_start && range > 4) {
			const uint8_t ops[8] = {
				0x41, // DW_CFA_advance_loc 1 (+4 bytes)
				0x0c, 29, 16, // DW_CFA_def_cfa x29, 16
				0x9d, 2, // DW_CFA_offset x29, -16
				0x9e, 1 // DW_CFA_offset x30, -8
			};
			out.insert(out.end(), ops, ops + 8);
		} else if (!is_symbol_start) {
			const uint8_t ops[8] = {
				0x0c, 29, 16, // DW_CFA_def_cfa x29, 16
				0x9d, 2, // DW_CFA_offset x29, -16
				0x9e, 1, // DW_CFA_offset x30, -8
				0x00 // DW_CFA_nop
			};
			out.insert(out.end(), ops, ops + 8);
		} else {
			for (int i = 0; i < 8; i++)
				out.push_back(0x00);
		}
	};

	std::vector<uint64_t> offs;
	offs.reserve(sorted_offsets.size());
	for (uint64_t off : sorted_offsets) {
		if (off < text_span)
			offs.push_back(off);
	}
	if (offs.empty()) {
		emit_fde(4, text_span, false);
	} else {
		if (offs.front() > 4)
			emit_fde(4, offs.front(), false);
		for (size_t i = 0; i < offs.size(); i++) {
			uint64_t s = offs[i];
			uint64_t e =
				(i + 1 < offs.size()) ? offs[i + 1] : text_span;
			emit_fde(s, e, s != 0);
		}
	}

	append_le32(out, 0); // zero-length terminator
	return out;
}

} // namespace

std::vector<uint8_t> build_stub_elf(const ModuleSymbols &module)
{
	ElfBuilder b;

	StrTab dynstr;
	StrTab strtab;
	StrTab shstr;

	std::vector<Elf64_Sym_D> dynsyms(1, Elf64_Sym_D{});
	std::vector<Elf64_Sym_D> symtab(1, Elf64_Sym_D{});
	// Mach-O nlist has no symbol size; derive a positive FUNC size from
	// the gap to the next symbol so uprobe boundary checks that reject
	// st_size==0 succeed. Stub addresses are never executed, so an
	// estimated span is safe while the probe offset stays in range.
	std::vector<uint64_t> sorted_offsets;
	sorted_offsets.reserve(module.symbols.size());
	for (const auto &s : module.symbols)
		sorted_offsets.push_back(s.offset);
	std::sort(sorted_offsets.begin(), sorted_offsets.end());
	sorted_offsets.erase(std::unique(sorted_offsets.begin(),
					 sorted_offsets.end()),
			     sorted_offsets.end());
	constexpr uint64_t default_func_size = 32;
	auto size_for = [&](uint64_t off) -> uint64_t {
		auto it = std::upper_bound(sorted_offsets.begin(),
					   sorted_offsets.end(), off);
		if (it == sorted_offsets.end()) {
			if (module.text_size > off)
				return std::max(default_func_size,
						module.text_size - off);
			return default_func_size;
		}
		uint64_t gap = *it - off;
		return gap > 0 ? gap : default_func_size;
	};
	uint64_t text_span = std::max<uint64_t>(16, module.text_size);
	for (const auto &s : module.symbols)
		text_span = std::max(text_span, s.offset + size_for(s.offset));
	for (const auto &n : module.sdt_notes)
		text_span = std::max(text_span, n.addr + 8);
	text_span = (text_span + 15u) & ~uint64_t(15);
	// Spans <= 1 MiB (normal Mach-O binaries) are physically backed; the
	// huge synthetic libSystem offset region is declared only via
	// sh_size/PT_LOAD.p_memsz (BSS-style zero tail) to avoid materializing
	// hundreds of MiB per stub.
	constexpr uint64_t inline_text_limit = 1u << 20;
	const bool physically_back_text = text_span <= inline_text_limit;
	for (const auto &s : module.symbols) {
		Elf64_Sym_D sym{};
		sym.st_name = dynstr.intern(s.name);
		sym.st_info = (STB_GLOBAL << 4) | STT_FUNC;
		sym.st_shndx = 1; // .text section index
		sym.st_value = s.offset;
		sym.st_size = size_for(s.offset);
		dynsyms.push_back(sym);
	}
	for (const auto &s : module.symbols) {
		Elf64_Sym_D sym{};
		sym.st_name = strtab.intern(s.name);
		sym.st_info = (STB_GLOBAL << 4) | STT_FUNC;
		sym.st_shndx = 1;
		sym.st_value = s.offset;
		sym.st_size = size_for(s.offset);
		symtab.push_back(sym);
	}
	auto note_bytes = build_stapsdt_notes(module.sdt_notes);
	const bool has_notes = !note_bytes.empty();
	const uint64_t eh_span = std::max<uint64_t>(
		16, module.text_size > 0 ?
			    ((module.text_size + 15u) & ~uint64_t(15)) :
			    std::min<uint64_t>(text_span, 64ULL << 20));
	auto eh_frame_bytes = build_arm64_eh_frame(sorted_offsets, eh_span);

	const uint32_t name_text = shstr.intern(".text");
	const uint32_t name_note = shstr.intern(".note.stapsdt");
	const uint32_t name_eh_frame = shstr.intern(".eh_frame");
	const uint32_t name_dynsym = shstr.intern(".dynsym");
	const uint32_t name_dynstr = shstr.intern(".dynstr");
	const uint32_t name_symtab = shstr.intern(".symtab");
	const uint32_t name_strtab = shstr.intern(".strtab");
	const uint32_t name_shstrtab = shstr.intern(".shstrtab");

	const size_t shnum = has_notes ? 9 : 8;
	const size_t shstrndx = shnum - 1;
	const size_t phnum = has_notes ? 2 : 1;

	Elf64_Ehdr_D eh{};
	static const uint8_t ident[16] = { 0x7f, 'E', 'L', 'F', 2, 1, 1, 0,
					   0,	 0,   0,   0,	0, 0, 0, 0 };
	memcpy(eh.e_ident, ident, 16);
	eh.e_type = ET_DYN;
	eh.e_machine = EM_AARCH64;
	eh.e_version = 1;
	eh.e_phoff = sizeof(Elf64_Ehdr_D);
	eh.e_shoff = 0; // patched later
	eh.e_ehsize = sizeof(Elf64_Ehdr_D);
	eh.e_phentsize = sizeof(Elf64_Phdr_D);
	eh.e_phnum = (uint16_t)phnum;
	eh.e_shentsize = sizeof(Elf64_Shdr_D);
	eh.e_shnum = (uint16_t)shnum;
	eh.e_shstrndx = (uint16_t)shstrndx;
	b.append_pod(eh);

	size_t ph_load_idx = b.buf.size();
	Elf64_Phdr_D ph_load{};
	ph_load.p_type = PT_LOAD_V;
	ph_load.p_flags = PF_R | PF_X;
	ph_load.p_align = 0x1000;
	b.append_pod(ph_load);
	size_t ph_note_idx = 0;
	if (has_notes) {
		ph_note_idx = b.buf.size();
		Elf64_Phdr_D ph_note{};
		ph_note.p_type = PT_NOTE_V;
		ph_note.p_flags = PF_R;
		ph_note.p_align = 4;
		b.append_pod(ph_note);
	}

	size_t text_off = b.buf.size();
	if (physically_back_text)
		b.buf.resize(text_off + (size_t)text_span, 0);
	else
		b.pad(16);

	size_t note_off = 0;
	if (has_notes) {
		b.pad(8);
		note_off = b.buf.size();
		b.append(note_bytes.data(), note_bytes.size());
	}

	b.pad(8);
	size_t eh_frame_off = b.buf.size();
	b.append(eh_frame_bytes.data(), eh_frame_bytes.size());

	b.pad(8);
	size_t dynsym_off = b.buf.size();
	for (const auto &s : dynsyms)
		b.append_pod(s);
	b.pad(1);
	size_t dynstr_off = b.buf.size();
	b.append(dynstr.bytes.data(), dynstr.bytes.size());

	b.pad(8);
	size_t symtab_off = b.buf.size();
	for (const auto &s : symtab)
		b.append_pod(s);
	b.pad(1);
	size_t strtab_off = b.buf.size();
	b.append(strtab.bytes.data(), strtab.bytes.size());

	b.pad(1);
	size_t shstrtab_off = b.buf.size();
	b.append(shstr.bytes.data(), shstr.bytes.size());

	b.pad(8);
	size_t shoff = b.buf.size();

	b.append_pod(Elf64_Shdr_D{}); // null section

	Elf64_Shdr_D sh_text{};
	sh_text.sh_name = name_text;
	sh_text.sh_type = SHT_PROGBITS;
	sh_text.sh_flags = SHF_ALLOC | SHF_EXECINSTR;
	sh_text.sh_addr = text_off;
	sh_text.sh_offset = text_off;
	sh_text.sh_size = text_span;
	sh_text.sh_addralign = 8;
	b.append_pod(sh_text);

	if (has_notes) {
		Elf64_Shdr_D sh_note{};
		sh_note.sh_name = name_note;
		sh_note.sh_type = 7; // SHT_NOTE
		sh_note.sh_offset = note_off;
		sh_note.sh_size = note_bytes.size();
		sh_note.sh_addralign = 4;
		b.append_pod(sh_note);
	}

	Elf64_Shdr_D sh_eh_frame{};
	sh_eh_frame.sh_name = name_eh_frame;
	sh_eh_frame.sh_type = SHT_PROGBITS;
	sh_eh_frame.sh_flags = SHF_ALLOC;
	sh_eh_frame.sh_addr = eh_frame_off;
	sh_eh_frame.sh_offset = eh_frame_off;
	sh_eh_frame.sh_size = eh_frame_bytes.size();
	sh_eh_frame.sh_addralign = 8;
	b.append_pod(sh_eh_frame);

	Elf64_Shdr_D sh_dynsym{};
	sh_dynsym.sh_name = name_dynsym;
	sh_dynsym.sh_type = SHT_DYNSYM;
	sh_dynsym.sh_flags = SHF_ALLOC;
	sh_dynsym.sh_addr = dynsym_off;
	sh_dynsym.sh_offset = dynsym_off;
	sh_dynsym.sh_size = dynsyms.size() * sizeof(Elf64_Sym_D);
	sh_dynsym.sh_addralign = 8;
	sh_dynsym.sh_entsize = sizeof(Elf64_Sym_D);
	b.append_pod(sh_dynsym);

	Elf64_Shdr_D sh_dynstr{};
	sh_dynstr.sh_name = name_dynstr;
	sh_dynstr.sh_type = SHT_STRTAB;
	sh_dynstr.sh_flags = SHF_ALLOC;
	sh_dynstr.sh_addr = dynstr_off;
	sh_dynstr.sh_offset = dynstr_off;
	sh_dynstr.sh_size = dynstr.bytes.size();
	b.append_pod(sh_dynstr);

	Elf64_Shdr_D sh_symtab{};
	sh_symtab.sh_name = name_symtab;
	sh_symtab.sh_type = SHT_SYMTAB;
	sh_symtab.sh_offset = symtab_off;
	sh_symtab.sh_size = symtab.size() * sizeof(Elf64_Sym_D);
	sh_symtab.sh_addralign = 8;
	sh_symtab.sh_entsize = sizeof(Elf64_Sym_D);
	b.append_pod(sh_symtab);

	Elf64_Shdr_D sh_strtab{};
	sh_strtab.sh_name = name_strtab;
	sh_strtab.sh_type = SHT_STRTAB;
	sh_strtab.sh_offset = strtab_off;
	sh_strtab.sh_size = strtab.bytes.size();
	b.append_pod(sh_strtab);

	Elf64_Shdr_D sh_shstr{};
	sh_shstr.sh_name = name_shstrtab;
	sh_shstr.sh_type = SHT_STRTAB;
	sh_shstr.sh_offset = shstrtab_off;
	sh_shstr.sh_size = shstr.bytes.size();
	b.append_pod(sh_shstr);

	// Section indices: 0 null, 1 .text, [2 .note], .eh_frame, then
	// dynsym/dynstr/symtab/strtab/shstrtab.
	size_t idx_dynsym = has_notes ? 4 : 3;
	size_t idx_dynstr = idx_dynsym + 1;
	size_t idx_symtab = idx_dynsym + 2;
	size_t idx_strtab = idx_dynsym + 3;
	auto *shdrs = reinterpret_cast<Elf64_Shdr_D *>(b.buf.data() + shoff);
	shdrs[idx_dynsym].sh_link = (uint32_t)idx_dynstr;
	shdrs[idx_dynsym].sh_info = 1; // first global symbol
	shdrs[idx_symtab].sh_link = (uint32_t)idx_strtab;
	shdrs[idx_symtab].sh_info = 1;

	b.patch_u64(offsetof(Elf64_Ehdr_D, e_shoff), shoff);
	b.patch_u64(ph_load_idx + offsetof(Elf64_Phdr_D, p_offset), 0);
	b.patch_u64(ph_load_idx + offsetof(Elf64_Phdr_D, p_vaddr), 0);
	b.patch_u64(ph_load_idx + offsetof(Elf64_Phdr_D, p_paddr), 0);
	b.patch_u64(ph_load_idx + offsetof(Elf64_Phdr_D, p_filesz),
		    b.buf.size());
	// Large synthetic stubs extend the segment with a BSS-style zero tail.
	const uint64_t span_page = (text_span + 0xfff) & ~uint64_t(0xfff);
	const uint64_t memsz =
		physically_back_text ?
			b.buf.size() :
			std::max<uint64_t>(b.buf.size(), span_page);
	b.patch_u64(ph_load_idx + offsetof(Elf64_Phdr_D, p_memsz), memsz);
	if (has_notes) {
		b.patch_u64(ph_note_idx + offsetof(Elf64_Phdr_D, p_offset),
			    note_off);
		b.patch_u64(ph_note_idx + offsetof(Elf64_Phdr_D, p_vaddr),
			    note_off);
		b.patch_u64(ph_note_idx + offsetof(Elf64_Phdr_D, p_paddr),
			    note_off);
		b.patch_u64(ph_note_idx + offsetof(Elf64_Phdr_D, p_filesz),
			    note_bytes.size());
		b.patch_u64(ph_note_idx + offsetof(Elf64_Phdr_D, p_memsz),
			    note_bytes.size());
	}
	return std::move(b.buf);
}

} // namespace darwin
} // namespace bpftime
