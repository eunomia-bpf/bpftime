#pragma once

#include "nv_attach_utils.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace bpftime::attach::elf_introspect {

struct loaded_module {
	std::string path;
	std::uintptr_t base_addr = 0;
};

struct symbol_range {
	std::uintptr_t start = 0;
	std::uintptr_t end = 0; // inclusive-exclusive, may equal start when unknown
	std::string name;
};

std::vector<loaded_module> list_loaded_modules();

std::optional<std::pair<const void *, std::size_t>>
find_section_in_memory(const loaded_module &mod, std::string_view section_name);

std::vector<const __fatBinC_Wrapper_t *>
scan_fatbin_wrappers(const void *section_addr, std::size_t section_size);

std::vector<symbol_range> read_function_symbols(const loaded_module &mod);

} // namespace bpftime::attach::elf_introspect
