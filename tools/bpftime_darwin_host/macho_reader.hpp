/* SPDX-License-Identifier: MIT
 *
 * Mach-O / dyld_shared_cache symbol extraction for the Darwin host daemon.
 */
#ifndef BPFTIME_DARWIN_MACHO_READER_HPP
#define BPFTIME_DARWIN_MACHO_READER_HPP

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bpftime
{
namespace darwin
{

struct SymbolEntry {
	std::string name; // ELF-style name (Mach-O leading '_' stripped)
	uint64_t offset = 0; // relative to the module's __TEXT base
};

struct StapSdtEntry {
	uint64_t addr = 0; // probe location, relative to __TEXT base
	std::string provider;
	std::string probe;
	std::string argfmt;
};

struct ModuleSymbols {
	std::string canonical_path;
	uint64_t text_vmaddr = 0;
	uint64_t text_size = 0;
	std::vector<SymbolEntry> symbols;
	std::vector<StapSdtEntry> sdt_notes;
};

struct DeepSymbol {
	std::string module;
	uint64_t offset = 0;
};

// Parse a thin 64-bit or Fat (ARM64 slice) Mach-O, including LC_SYMTAB and
// __TEXT,__dof_* / __DATA,__dof USDT sections.
ModuleSymbols parse_disk_macho(const std::string &path);

// Enumerate exported functions of a loaded image through Frida Gum, used
// for dyld_shared_cache libraries that have no standalone file.
ModuleSymbols enumerate_runtime_exports(const std::string &real_path);

ModuleSymbols enumerate_runtime_symbols(const std::string &real_path);

// Resolve a dyld_shared_cache re-export veneer to its deep implementation.
DeepSymbol resolve_deep_symbol(const std::string &real_module,
			       const std::vector<std::string> &candidates);

// Virtual /bpftime-dyld paths for dyld_shared_cache libraries.
bool is_dyld_virtual_path(const std::string &path);
std::string dyld_virtual_to_real(const std::string &path);

constexpr uint64_t UMBRELLA_SYNTH_BASE = 0x10000000ULL;
constexpr uint64_t UMBRELLA_SYNTH_STRIDE = 16;

struct UmbrellaIndex {
	// synthetic offset -> (real child dylib path, real child offset)
	std::unordered_map<uint64_t, std::pair<std::string, uint64_t>>
		translations;
	// veneer symbol name -> deep hookable implementation
	std::unordered_map<std::string, DeepSymbol> deep_symbols;
	ModuleSymbols stub;
};

// Aggregate the libSystem.B.dylib umbrella re-exports.
UmbrellaIndex build_libsystem_umbrella(const std::string &virtual_path);

} // namespace darwin
} // namespace bpftime

#endif
