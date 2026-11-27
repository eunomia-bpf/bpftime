/* SPDX-License-Identifier: MIT
 *
 * Synthesizes in-memory 64-bit ARM64 ET_DYN stub ELF files from Mach-O
 * symbol tables so unmodified libbpf/bcc/bpftrace tooling can parse probe
 * offsets with libelf exactly as on Linux.
 */
#ifndef BPFTIME_REMOTE_ELF_SYNTH_HPP
#define BPFTIME_REMOTE_ELF_SYNTH_HPP

#include "macho_reader.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace bpftime
{
namespace remote
{

// Build a valid EM_AARCH64 ET_DYN ELF image containing both .dynsym and
// .symtab (plus .strtab/.dynstr/.shstrtab) and, when USDT probes exist, a
// .note.stapsdt section. A single PT_LOAD covers the whole file with
// p_vaddr=0, p_offset=0 so st_value is directly the module-relative offset.
std::vector<uint8_t> build_stub_elf(const ModuleSymbols &module);

} // namespace remote
} // namespace bpftime

#endif
