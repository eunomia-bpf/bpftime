#pragma once

#include <optional>
#include <string>
#include <vector>
#include <ebpf_inst.h>
#include "json.hpp"

namespace bpftime::attach
{

struct PtxPassSpec {
	std::string exec;
	std::string config;
};

struct PtxPassesConfig {
	std::vector<PtxPassSpec> passes;
};

// nlohmann::json ADL conversions
inline void from_json(const nlohmann::json &j, PtxPassSpec &s)
{
	j.at("exec").get_to(s.exec);
	if (j.contains("config"))
		j.at("config").get_to(s.config);
	else
		s.config = ""; // Empty config means use pass's built-in config
}

inline void from_json(const nlohmann::json &j, PtxPassesConfig &cfg)
{
	if (j.contains("passes"))
		cfg.passes = j.at("passes").get<std::vector<PtxPassSpec>>();
}

// Load pipeline config from JSON file.
PtxPassesConfig load_passes_config(const std::string &path);
// Load from env (BPFTIME_PTXPASS_CONFIG) or default path
PtxPassesConfig
load_passes_config_from_env_or_default(const std::string &default_path);

// Load passes from a directory containing multiple JSON files with fields
// { "exec": "/abs/path/to/executable", "config": "/abs/or/rel/config.json" }
PtxPassesConfig load_passes_from_directory(const std::string &dir);

// Prefer directory from env (BPFTIME_PTXPASS_DIR). If found passes, return
// them; otherwise, fallback to env config or default config path
PtxPassesConfig
load_passes_from_envdir_or_default(const std::string &default_dir,
				   const std::string &default_config_path);

// Run PTX through all passes for a given attach point.
std::optional<std::string> run_ptx_pipeline(const std::string &attach_point,
					    const std::string &original_ptx,
					    const PtxPassesConfig &config);

// Default directory for pass definition JSONs (can be overridden at build time)
#ifndef DEFAULT_PASSES_DIR
#define DEFAULT_PASSES_DIR "attach/nv_attach_impl/pass"
#endif

struct PassAttachPointSpec {
	int type;
	std::string expected_func_name_regex;
	// Optional: pass-config style attach points
	std::vector<std::string> includes;
	std::vector<std::string> excludes;
};

struct PassDefinition {
	std::string executable;
	PassAttachPointSpec attach_point;
};

inline void from_json(const nlohmann::json &j, PassAttachPointSpec &ap)
{
	if (j.contains("type"))
		j.at("type").get_to(ap.type);
	if (j.contains("expected_func_name_regex"))
		j.at("expected_func_name_regex")
			.get_to(ap.expected_func_name_regex);
	if (j.contains("includes"))
		ap.includes = j.at("includes").get<std::vector<std::string>>();
	if (j.contains("excludes"))
		ap.excludes = j.at("excludes").get<std::vector<std::string>>();
}

inline void from_json(const nlohmann::json &j, PassDefinition &d)
{
	j.at("executable").get_to(d.executable);
	if (j.contains("attach_point"))
		d.attach_point =
			j.at("attach_point").get<PassAttachPointSpec>();
	if (j.contains("attach_points"))
		d.attach_point =
			j.at("attach_points").get<PassAttachPointSpec>();
}

// Load pass definitions from a directory (each file is a JSON definition)
std::vector<PassDefinition>
load_pass_definitions_from_dir(const std::string &dir);

// Run a single pass executable with JSON stdin/stdout.
// Input JSON fields:
//  - full_ptx
//  - to_patch_kernel
//  - global_ebpf_map_info_symbol
//  - ebpf_communication_data_symbol
// Output JSON fields:
//  - output_ptx (may be empty or omitted to represent no-change)
std::optional<std::string> run_pass_executable_json(
	const std::string &exec, const std::string &full_ptx,
	const std::string &to_patch_kernel, const std::string &map_sym,
	const std::string &const_sym, const std::vector<ebpf_inst> &ebpf_insts);

} // namespace bpftime::attach
