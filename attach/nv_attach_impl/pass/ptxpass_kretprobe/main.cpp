#include "ptxpass/core.hpp"
#include <vector>
#include <exception>
#include <iostream>
#include <string>

static ptxpass::PassConfig GetDefaultConfig()
{
	ptxpass::PassConfig cfg;
	cfg.name = "kretprobe";
	cfg.description = "Inject call before ret for kretprobe/*";
	cfg.attachPoints.includes = { "^kretprobe/.*$" };
	cfg.attachPoints.excludes = {};
	cfg.parameters = nlohmann::json::object();
	cfg.validation = nlohmann::json::object();
	return cfg;
}

static std::pair<std::string, bool>
PatchRetprobe(const std::string &ptx, const std::string &kernel,
	       const std::vector<uint64_t> &ebpf_words)
{
	auto func_ptx =
		ptxpass::compile_ebpf_to_ptx_from_words(ebpf_words, "sm_60");
	auto body = ptxpass::find_kernel_body(ptx, kernel);
	if (body.first == std::string::npos)
		return { "", false };
	std::string out = ptx;
	std::string section = out.substr(body.first, body.second - body.first);
	static std::regex retpat(R"((\s+)(ret;))");
	std::string fname = std::string("__probe_func__") + kernel;
	std::ostringstream def;
	def << ".func " << fname << "\n" << func_ptx << "\n";
	section = std::regex_replace(
		section, retpat, std::string("$1call ") + fname + ";\n$1$2");
	out.replace(body.first, body.second - body.first, section);
	out = def.str() + "\n" + out;
	ptxpass::log_transform_stats("kretprobe", 1, ptx.size(), out.size());
	return { out, true };
}

static void PrintUsage(const char *argv0)
{
	std::cerr
		<< "Usage: " << argv0
		<< " [--config <path>|--config] [--print-config] [--log-level <level>] [--dry-run]\n";
}

int main(int argc, char **argv)
{
	using namespace ptxpass;
	std::string configPath;
	bool dryRun = false;
	bool printConfigOnly = false;

	for (int i = 1; i < argc; ++i) {
		std::string a = argv[i];
		if (a == "--config") {
			if (i + 1 < argc && argv[i + 1][0] != '-') {
				configPath = argv[++i];
			} else {
				printConfigOnly = true;
			}
		} else if (a == "--print-config") {
			printConfigOnly = true;
		} else if (a == "--dry-run") {
			dryRun = true;
		} else if (a == "--help" || a == "-h") {
            PrintUsage(argv[0]);
			return ExitCode::Success;
		} else if (a == "--log-level") {
			// Ignored in minimal skeleton
			if (i + 1 < argc)
				++i;
		} else {
			// Ignore unknown for minimal version
		}
	}

	try {
		PassConfig cfg;
		if (printConfigOnly) {
            cfg = GetDefaultConfig();
			nlohmann::json j;
			j["name"] = cfg.name;
			j["description"] = cfg.description;
			j["attach_points"]["includes"] =
				cfg.attachPoints.includes;
			j["attach_points"]["excludes"] =
				cfg.attachPoints.excludes;
			j["parameters"] = cfg.parameters;
			j["validation"] = cfg.validation;
			std::cout << j.dump(4);
			return ExitCode::Success;
		}
		if (!configPath.empty()) {
			cfg = JsonConfigLoader::load_from_file(configPath);
		} else {
			cfg = GetDefaultConfig();
		}
		auto matcher = AttachPointMatcher(cfg.attachPoints);
		auto ap = get_env("PTX_ATTACH_POINT");
		if (ap.empty()) {
			std::cerr << "PTX_ATTACH_POINT is not set\n";
			return ExitCode::ConfigError;
		}

		std::string stdinData = read_all_from_stdin();
		auto [rr, isJson] = parse_runtime_request(stdinData);
		if (!isJson)
			return ExitCode::InputError;
		if (!matcher.matches(ap))
			return ExitCode::Success;
		if (dryRun)
			return ExitCode::Success;
		if (!validate_input(rr.input.full_ptx, cfg.validation))
			return ExitCode::TransformFailed;
		auto [out, modified] = PatchRetprobe(rr.input.full_ptx,
						      rr.input.to_patch_kernel,
						      rr.ebpf_instructions);
		if (modified && !is_whitespace_only(out))
			emit_runtime_output(out);
		return ExitCode::Success;
	} catch (const std::runtime_error &e) {
		std::cerr << e.what() << "\n";
		return ExitCode::ConfigError;
	} catch (const std::exception &e) {
		std::cerr << e.what() << "\n";
		return ExitCode::UnknownError;
	} catch (...) {
		std::cerr << "Unknown error\n";
		return ExitCode::UnknownError;
	}
}
