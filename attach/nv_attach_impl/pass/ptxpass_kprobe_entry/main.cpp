#include "ptxpass/core.hpp"
#include <exception>
#include <iostream>
#include <string>

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
        if (!isJson) return ExitCode::InputError;
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
		auto [out, modified] =
			instrumentEntry(ri.full_ptx, cfg.parameters);
        if (modified && !isWhitespaceOnly(out)) emitRuntimeOutput(out);
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
