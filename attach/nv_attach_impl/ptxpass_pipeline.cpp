#include "ptxpass_pipeline.hpp"
#include <boost/process.hpp>
#include <fstream>
#include "json.hpp"
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>

namespace bpftime::attach
{

using nlohmann::json;

PtxPassesConfig load_passes_config(const std::string &path)
{
	std::ifstream ifs(path);
	if (!ifs.is_open()) {
		throw std::runtime_error("Cannot open passes config: " + path);
	}
	json j;
	ifs >> j;
	PtxPassesConfig cfg;
	if (j.contains("fail_fast"))
		cfg.fail_fast = j["fail_fast"].get<bool>();
	if (j.contains("passes") && j["passes"].is_array()) {
		for (auto &p : j["passes"]) {
			PtxPassSpec spec;
			spec.exec = p["exec"].get<std::string>();
			spec.config = p["config"].get<std::string>();
			cfg.passes.push_back(std::move(spec));
		}
	}
	return cfg;
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
	child c(exec + " --config " + config, std_out > child_stdout,
		std_err > child_stderr, std_in < child_stdin, env);
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
		auto res = run_pass_executable_json(pass.exec, current,
						    to_patch_kernel, map_sym,
						    const_sym);
		if (!res.has_value()) {
			if (config.fail_fast)
				return std::nullopt;
			else
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
			PtxPassSpec spec;
			spec.exec = j.at("exec").get<std::string>();
			spec.config = j.at("config").get<std::string>();
			cfg.passes.push_back(std::move(spec));
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
			PassDefinition d;
			std::string exec =
				j.at("executable").get<std::string>();
			if (!exec.empty() && !fs::path(exec).is_absolute()) {
				// Resolve relative to the JSON file directory
				exec = (entry.path().parent_path() / exec)
					       .string();
			}
			d.executable = exec;
			d.priority = j.at("priority").get<int>();
			auto &ap = j.at("attach_point");
			d.attach_point.type = ap.at("type").get<int>();
			d.attach_point.expected_func_name_regex =
				ap.at("expected_func_name_regex")
					.get<std::string>();
			defs.push_back(std::move(d));
		} catch (const std::exception &e) {
			// skip invalid files silently; caller can log aggregate
			continue;
		}
	}
	std::sort(defs.begin(), defs.end(),
		  [](const PassDefinition &a, const PassDefinition &b) {
			  if (a.priority != b.priority)
				  return a.priority < b.priority;
			  return a.executable < b.executable; // stable
							      // deterministic
							      // order
		  });
	return defs;
}

std::optional<std::string>
run_pass_executable_json(const std::string &exec, const std::string &full_ptx,
			 const std::string &to_patch_kernel,
			 const std::string &map_sym,
			 const std::string &const_sym)
{
	using namespace boost::process;
	ipstream child_stdout;
	ipstream child_stderr;
	opstream child_stdin;
	child c(exec, std_out > child_stdout, std_err > child_stderr,
		std_in < child_stdin);
	json in = { { "full_ptx", full_ptx },
		    { "to_patch_kernel", to_patch_kernel },
		    { "global_ebpf_map_info_symbol", map_sym },
		    { "ebpf_communication_data_symbol", const_sym } };
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
		auto j = json::parse(out_str);
		if (j.contains("output_ptx") && j["output_ptx"].is_string())
			return j["output_ptx"].get<std::string>();
		return std::string();
	} catch (...) {
		return std::nullopt; // invalid json
	}
}

} // namespace bpftime::attach
