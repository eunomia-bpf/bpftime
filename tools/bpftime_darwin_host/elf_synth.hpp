/* SPDX-License-Identifier: MIT
 *
 * Synthesizes in-memory 64-bit ARM64 ET_DYN stub ELF files from Mach-O
 * symbol tables so unmodified libbpf/bcc/bpftrace tooling parses probe
 * offsets with libelf exactly as on Linux.
 */
#ifndef BPFTIME_DARWIN_ELF_SYNTH_HPP
#define BPFTIME_DARWIN_ELF_SYNTH_HPP

#include "macho_reader.hpp"

#include <string>
#include <vector>

namespace bpftime
{
namespace darwin
{

// EM_AARCH64 ET_DYN image with .dynsym/.symtab (plus string tables), an
// optional .note.stapsdt section and an ARM64 .eh_frame section. One PT_LOAD
// covers the whole file at p_vaddr=0 so st_value is the module-relative offset.
std::vector<uint8_t> build_stub_elf(const ModuleSymbols &module);

} // namespace darwin
} // namespace bpftime

#endif
