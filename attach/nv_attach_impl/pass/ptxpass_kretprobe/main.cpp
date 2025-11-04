#include "json.hpp"
#include "ptxpass/core.hpp"
#include <cstdio>
#include <cstring>
#include <ostream>
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
	if (body.first == std::string::npos) {
		return { ptx, false };
	}
	std::string out = ptx;
	std::string section = out.substr(body.first, body.second - body.first);
	static std::regex retpat(R"((\s*)(ret;))");
	section = std::regex_replace(
		section, retpat, std::string("$1call ") + fname + ";\n$1$2");
	out.replace(body.first, body.second - body.first, section);
	out = func_ptx + "\n" + out;
	ptxpass::log_transform_stats("kretprobe", 1, ptx.size(), out.size());
	return { out, true };
}

extern "C" void print_config(int length, char *out)
{
	auto cfg = get_default_config();
	nlohmann::json output_json;
	ptxpass::pass_config::to_json(output_json, cfg);
	snprintf(out, length, "%s", output_json.dump().c_str());
}

extern "C" int process_input(const char *input, int length, char *output)
{
	using namespace ptxpass;
	try {
		auto cfg = get_default_config();
		auto matcher = AttachPointMatcher(cfg.attach_points);

		auto runtime_request = pass_runtime_request_from_string(input);
		if (!validate_input(runtime_request.input.full_ptx,
				    cfg.validation))
			return ExitCode::TransformFailed;
		auto [out, modified] = patch_retprobe(
			runtime_request.input.full_ptx,
			runtime_request.input.to_patch_kernel,
			runtime_request.get_uint64_ebpf_instructions());
		snprintf(output, length, "%s",
			 emit_runtime_response_and_return(out).c_str());
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
