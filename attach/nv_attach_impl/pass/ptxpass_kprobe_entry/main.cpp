#include "ptxpass/core.hpp"
#include <vector>
#include <exception>
#include <iostream>
#include <string>

static std::pair<std::string, bool>
patch_entry(const std::string &ptx, const std::string &kernel,
	    const std::vector<uint64_t> &ebpf_words)
{
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

static void print_usage(const char *argv0)
{
	std::cerr << "Usage: " << argv0
		  << " --config <path> [--log-level <level>] [--dry-run]\n";
}

int main(int argc, char **argv)
{
	using namespace ptxpass;
	std::string configPath;
	bool dryRun = false;

	for (int i = 1; i < argc; ++i) {
		std::string a = argv[i];
		if (a == "--config" && i + 1 < argc) {
			configPath = argv[++i];
		} else if (a == "--dry-run") {
			dryRun = true;
		} else if (a == "--help" || a == "-h") {
			print_usage(argv[0]);
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
		if (configPath.empty()) {
			std::cerr << "Missing --config\n";
			return ExitCode::ConfigError;
		}
		auto cfg = JsonConfigLoader::loadFromFile(configPath);
		auto matcher = AttachPointMatcher(cfg.attachPoints);
		auto ap = getEnv("PTX_ATTACH_POINT");
		if (ap.empty()) {
			std::cerr << "PTX_ATTACH_POINT is not set\n";
			return ExitCode::ConfigError;
		}

		// JSON-only
		std::string stdinData = readAllFromStdin();
		auto [ri, isJson] = parseRuntimeInput(stdinData);
		if (!isJson)
			return ExitCode::InputError;
		if (!matcher.matches(ap)) {
			// Not matched: empty output
			return ExitCode::Success;
		}
		if (dryRun) {
			return ExitCode::Success;
		}
		if (!validateInput(ri.full_ptx, cfg.validation)) {
			return ExitCode::TransformFailed;
		}
		std::vector<uint64_t> words;
		try {
			auto j = nlohmann::json::parse(stdinData);
			if (j.contains("ebpf_instructions"))
				words = j["ebpf_instructions"]
						.get<std::vector<uint64_t>>();
		} catch (...) {
		}
		auto [out, modified] =
			patch_entry(ri.full_ptx, ri.to_patch_kernel, words);
		if (modified && !isWhitespaceOnly(out))
			emitRuntimeOutput(out);
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
