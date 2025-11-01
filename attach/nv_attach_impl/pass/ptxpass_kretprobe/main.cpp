#include "json.hpp"
#include "ptxpass/core.hpp"
#include <vector>
#include <exception>
#include <iostream>
#include <string>

namespace retprobe_params
{
struct RetprobeParams {
	std::string save_strategy = "minimal"; // "minimal" or "full"
	bool emit_nops_for_alignment = false;
	int pad_nops = 1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(RetprobeParams, save_strategy,
						emit_nops_for_alignment,
						pad_nops);
} // namespace retprobe_params

static ptxpass::pass_config::PassConfig get_default_config()
{
	ptxpass::pass_config::PassConfig cfg;
	cfg.name = "kretprobe";
	cfg.description = "Inject call before ret for kretprobe/*";
	cfg.attach_points.includes = { "^kretprobe/.*$" };
	cfg.attach_points.excludes = {};
	cfg.parameters = nlohmann::json::object();
	cfg.validation = nlohmann::json::object();
	cfg.attach_type = 9; // kretprobe
	return cfg;
}

static std::pair<std::string, bool>
patch_retprobe(const std::string &ptx, const std::string &kernel,
	       const std::vector<uint64_t> &ebpf_words)
{
	std::string fname = std::string("__retprobe_func__") + kernel;

	auto func_ptx = ptxpass::compile_ebpf_to_ptx_from_words(
		ebpf_words, "sm_60", fname, true, false);
	auto body = ptxpass::find_kernel_body(ptx, kernel);
	if (body.first == std::string::npos)
		return { "", false };
	std::string out = ptx;
	std::string section = out.substr(body.first, body.second - body.first);
	static std::regex retpat(R"((\s+)(ret;))");
	section = std::regex_replace(
		section, retpat, std::string("$1call ") + fname + ";\n$1$2");
	out.replace(body.first, body.second - body.first, section);
	out = func_ptx + "\n" + out;
	ptxpass::log_transform_stats("kretprobe", 1, ptx.size(), out.size());
	return { out, true };
}

static void print_usage(const char *argv0)
{
	std::cerr
		<< "Usage: " << argv0
		<< " [--config <path>|--config] [--print-config] [--log-level <level>] [--dry-run]\n";
}

int main(int argc, char **argv)
{
	using namespace ptxpass;
	std::string config_path;
	bool dry_run = false;
	bool print_config_only = false;

	for (int i = 1; i < argc; ++i) {
		std::string a = argv[i];
		if (a == "--config") {
			if (i + 1 < argc && argv[i + 1][0] != '-') {
				config_path = argv[++i];
			} else {
				print_config_only = true;
			}
		} else if (a == "--print-config") {
			print_config_only = true;
		} else if (a == "--dry-run") {
			dry_run = true;
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
		pass_config::PassConfig cfg;
		if (print_config_only) {
			cfg = get_default_config();
			nlohmann::json output_json;
			pass_config::to_json(output_json, cfg);
			std::cout << output_json.dump(4);
			return ExitCode::Success;
		}
		if (!config_path.empty()) {
			cfg = load_pass_config_from_file(config_path);
		} else {
			cfg = get_default_config();
		}
		auto matcher = AttachPointMatcher(cfg.attach_points);

		std::string stdin_data = read_all_from_stdin();
		auto runtime_request =
			pass_runtime_request_from_string(stdin_data);
		if (dry_run)
			return ExitCode::Success;
		if (!validate_input(runtime_request.input.full_ptx,
				    cfg.validation))
			return ExitCode::TransformFailed;
		auto [out, modified] =
			patch_retprobe(runtime_request.input.full_ptx,
				       runtime_request.input.to_patch_kernel,
				       runtime_request.ebpf_instructions);
		if (modified && !is_whitespace_only(out))
			emit_runtime_response_and_print(out);
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
