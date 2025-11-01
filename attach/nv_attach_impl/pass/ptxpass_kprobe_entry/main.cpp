#include "ptxpass/core.hpp"
#include <vector>
#include <exception>
#include <iostream>
#include <string>

static ptxpass::PassConfig GetDefaultConfig()
{
	ptxpass::PassConfig cfg;
	cfg.name = "kprobe_entry";
	cfg.description =
		"Instrument PTX at kprobe entry points, excluding __memcapture";
	cfg.attachPoints.includes = { "^kprobe/.*$" };
	cfg.attachPoints.excludes = { "^kprobe/__memcapture$" };
	cfg.parameters = nlohmann::json{ { "insert_globaltimer", true } };
	return cfg;
}

static std::pair<std::string, bool>
PatchEntry(const std::string &ptx, const std::string &kernel,
	   const std::vector<uint64_t> &ebpf_words)
{
	if (ebpf_words.empty()) {
		return { ptx, false };
	}
	auto func_ptx =
		ptxpass::compile_ebpf_to_ptx_from_words(ebpf_words, "sm_60");
	auto body = ptxpass::find_kernel_body(ptx, kernel);
	if (body.first == std::string::npos)
		return { "", false };
	std::string out = ptx;
	size_t brace = out.find('{', body.first);
	if (brace == std::string::npos)
		return { "", false };
	size_t insertPos = brace + 1;
	if (insertPos < out.size() && out[insertPos] == '\n')
		insertPos++;
	std::string fname = std::string("__probe_func__") + kernel;
	std::ostringstream def;
	def << ".func " << fname << "\n" << func_ptx << "\n";
	out.insert(insertPos, std::string("\n    call ") + fname + ";\n");
	out = def.str() + "\n" + out;
	ptxpass::log_transform_stats("kprobe_entry", 1, ptx.size(), out.size());
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
		if (!matcher.matches(ap)) {
			// Not matched: empty output
			return ExitCode::Success;
		}
		if (dryRun) {
			return ExitCode::Success;
		}
		if (!validate_input(rr.input.full_ptx, cfg.validation)) {
			return ExitCode::TransformFailed;
		}
		auto [out, modified] =
			PatchEntry(rr.input.full_ptx, rr.input.to_patch_kernel,
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
