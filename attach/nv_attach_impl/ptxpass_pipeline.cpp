#include "ptxpass_pipeline.hpp"
#include <boost/process.hpp>
#include <fstream>
#include "json.hpp"
#include "ptxpass/core.hpp"
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

namespace bpftime::attach
{

using nlohmann::json;

// Use ptxpass::RuntimeInput for JSON payload; attach ebpf_instructions
// separately

PtxPassesConfig load_passes_config(const std::string &path)
{
	std::ifstream ifs(path);
	if (!ifs.is_open()) {
		throw std::runtime_error("Cannot open passes config: " + path);
	}
	json j;
	ifs >> j;
	return j.get<PtxPassesConfig>();
}

PtxPassesConfig
load_passes_config_from_env_or_default(const std::string &default_path)
{
	const char *env = std::getenv("BPFTIME_PTXPASS_CONFIG");
	if (env && *env) {
		return load_passes_config(env);
	}
	return load_passes_config(default_path);
}

static std::tuple<int, std::string, std::string>
run_single_pass(const std::string &exec, const std::string &config,
		const std::string &attach_point, const std::string &input)
{
	using namespace boost::process;
	ipstream child_stdout;
	ipstream child_stderr;
	opstream child_stdin;
	boost::process::environment env = boost::this_process::environment();
	env["PTX_ATTACH_POINT"] = attach_point;
	std::string cmd = exec;
	if (!config.empty())
		cmd += " --config " + config;
	child c(cmd, std_out > child_stdout, std_err > child_stderr,
		std_in < child_stdin, env);
	child_stdin << input;
	child_stdin.flush();
	child_stdin.pipe().close();
	std::ostringstream out, err;
	std::string line;
	while (child_stdout && std::getline(child_stdout, line)) {
		out << line << '\n';
	}
	while (child_stderr && std::getline(child_stderr, line)) {
		err << line << '\n';
	}
	c.wait();
	int code = c.exit_code();
	return { code, out.str(), err.str() };
}

std::optional<std::string> run_ptx_pipeline(const std::string &attach_point,
					    const std::string &original_ptx,
					    const PtxPassesConfig &config)
{
	std::string current = original_ptx;
	// Derive kernel name from attach_point, e.g. "kprobe/<kernel>" or
	// "kretprobe/<kernel>"
	std::string to_patch_kernel;
	{
		const std::string kp = "kprobe/";
		const std::string kr = "kretprobe/";
		if (attach_point.rfind(kp, 0) == 0) {
			to_patch_kernel = attach_point.substr(kp.size());
		} else if (attach_point.rfind(kr, 0) == 0) {
			to_patch_kernel = attach_point.substr(kr.size());
		}
	}
	const std::string map_sym = "map_info";
	const std::string const_sym = "constData";
	for (const auto &pass : config.passes) {
		std::vector<ebpf_inst> empty;
		auto res = run_pass_executable_json(pass.exec, current,
						    to_patch_kernel, map_sym,
						    const_sym, empty);
		if (!res.has_value()) {
			continue;
		}
		if (!res->empty())
			current = *res;
	}
	return current;
}

PtxPassesConfig load_passes_from_directory(const std::string &dir)
{
	PtxPassesConfig cfg;
	if (!std::filesystem::exists(dir))
		return cfg;
	for (auto &entry : std::filesystem::directory_iterator(dir)) {
		if (!entry.is_regular_file())
			continue;
		if (entry.path().extension() != ".json")
			continue;
		try {
			std::ifstream ifs(entry.path());
			json j;
			ifs >> j;
			cfg.passes.push_back(j.get<PtxPassSpec>());
		} catch (...) {
			// ignore broken file
		}
	}
	return cfg;
}

PtxPassesConfig
load_passes_from_envdir_or_default(const std::string &default_dir,
				   const std::string &default_config_path)
{
	const char *dir = std::getenv("BPFTIME_PTXPASS_DIR");
	if (dir && *dir) {
		auto cfg = load_passes_from_directory(dir);
		if (!cfg.passes.empty())
			return cfg;
	} else {
		auto cfg = load_passes_from_directory(default_dir);
		if (!cfg.passes.empty())
			return cfg;
	}
	return load_passes_config_from_env_or_default(default_config_path);
}

static inline bool is_nonempty_nonblank(const std::string &s)
{
	for (char c : s) {
		if (!(c == ' ' || c == '\n' || c == '\r' || c == '\t' ||
		      c == '\v' || c == '\f'))
			return true;
	}
	return false;
}

std::vector<PassDefinition>
load_pass_definitions_from_dir(const std::string &dir)
{
	std::vector<PassDefinition> defs;
	namespace fs = std::filesystem;
	if (!fs::exists(dir) || !fs::is_directory(dir))
		return defs;
	for (const auto &entry : fs::directory_iterator(dir)) {
		if (!entry.is_regular_file())
			continue;
		if (entry.path().extension() != ".json")
			continue;
		try {
			std::ifstream ifs(entry.path());
			json j;
			ifs >> j;
			PassDefinition d = j.get<PassDefinition>();
			if (!d.executable.empty() &&
			    !fs::path(d.executable).is_absolute()) {
				d.executable = (entry.path().parent_path() /
						d.executable)
						       .string();
			}
			defs.push_back(std::move(d));
		} catch (const std::exception &e) {
			// skip invalid files silently; caller can log aggregate
			continue;
		}
	}
	std::sort(defs.begin(), defs.end(),
		  [](const PassDefinition &a, const PassDefinition &b) {
			  return a.executable < b.executable; // stable
							      // deterministic
							      // order
		  });
	return defs;
}

std::optional<std::string> run_pass_executable_json(
	const std::string &exec, const std::string &full_ptx,
	const std::string &to_patch_kernel, const std::string &map_sym,
	const std::string &const_sym, const std::vector<ebpf_inst> &ebpf_insts)
{
	using namespace boost::process;
	ipstream child_stdout;
	ipstream child_stderr;
	opstream child_stdin;
	child c(exec, std_out > child_stdout, std_err > child_stderr,
		std_in < child_stdin);
	std::vector<uint64_t> words;
	if (!ebpf_insts.empty()) {
		const uint64_t *words_ptr =
			reinterpret_cast<const uint64_t *>(ebpf_insts.data());
		words.assign(words_ptr, words_ptr + ebpf_insts.size());
	}
	ptxpass::RuntimeInput ri;
	ri.full_ptx = full_ptx;
	ri.to_patch_kernel = to_patch_kernel;
	ri.global_ebpf_map_info_symbol = map_sym;
	ri.ebpf_communication_data_symbol = const_sym;
	nlohmann::json in = ri;
	if (!words.empty())
		in["ebpf_instructions"] = words;
	child_stdin << in.dump();
	child_stdin.flush();
	child_stdin.pipe().close();
	std::ostringstream out, err;
	std::string line;
	while (child_stdout && std::getline(child_stdout, line))
		out << line << '\n';
	while (child_stderr && std::getline(child_stderr, line))
		err << line << '\n';
	c.wait();
	if (c.exit_code() != 0)
		return std::nullopt;
	const std::string out_str = out.str();
	if (!is_nonempty_nonblank(out_str))
		return std::string();
	try {
		auto j = nlohmann::json::parse(out_str);
		ptxpass::RuntimeOutput ro = j.get<ptxpass::RuntimeOutput>();
		return ro.output_ptx;
	} catch (...) {
		return std::nullopt; // invalid json
	}
}

} // namespace bpftime::attach
