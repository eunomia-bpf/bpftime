/* SPDX-License-Identifier: MIT
 *
 * Mach-O / dyld_shared_cache symbol extraction for the Darwin host daemon.
 */
#ifndef BPFTIME_REMOTE_MACHO_READER_HPP
#define BPFTIME_REMOTE_MACHO_READER_HPP

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace bpftime
{
namespace remote
{

struct SymbolEntry {
	std::string name; // ELF-style name (Mach-O leading '_' stripped)
	uint64_t offset; // relative to the module's __TEXT base
};

struct StapSdtEntry {
	uint64_t addr; // probe location, relative to __TEXT base
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

ModuleSymbols enumerate_runtime_symbols(const std::string &real_path);

// Resolve a dyld_shared_cache re-export veneer (e.g. malloc) to the deep
// allocator implementation that calls branch to after lazy binding.
struct DeepSymbol {
	std::string module;
	uint64_t offset = 0;
};
DeepSymbol resolve_deep_symbol(const std::string &real_module,
			       const std::vector<std::string> &candidates);

// Virtual /bpftime-dyld path handling for dyld_shared_cache libraries.
bool is_dyld_virtual_path(const std::string &path);
std::string dyld_virtual_to_real(const std::string &path);
std::string dyld_real_to_virtual(const std::string &path);

// Parse a real on-disk Mach-O file (thin 64-bit or Fat universal, ARM64
// slice selected), including LC_SYMTAB and any __DATA,__dof USDT section.
ModuleSymbols parse_disk_macho(const std::string &path);

// Enumerate exported functions of a loaded image through Frida Gum,
// returning offsets relative to the module's runtime base. Used for
// dyld_shared_cache libraries that have no standalone file.
ModuleSymbols enumerate_runtime_exports(const std::string &real_path);

// Deterministic synthetic offset region used when aggregating the
// libSystem.B.dylib umbrella re-exports.
constexpr uint64_t UMBRELLA_SYNTH_BASE = 0x10000000ULL;
constexpr uint64_t UMBRELLA_SYNTH_STRIDE = 16;

struct UmbrellaIndex {
	// synthetic offset -> (real child dylib path, real child offset)
	std::unordered_map<uint64_t, std::pair<std::string, uint64_t>>
		translations;
	// veneer symbol name (e.g. "malloc") -> deep hookable implementation
	std::unordered_map<std::string, DeepSymbol> deep_symbols;
	ModuleSymbols stub;
};

// Build the aggregated libSystem.B.dylib stub symbols and the translation
// table consumed by perf_event_open.
UmbrellaIndex build_libsystem_umbrella(const std::string &virtual_path);

} // namespace remote
} // namespace bpftime

#endif
