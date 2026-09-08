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
	cfg.description = "Inject call before ret/exit for kretprobe/*";
	cfg.attach_points.includes = { "^kretprobe/.*$" };
	cfg.attach_points.excludes = {};
	cfg.parameters = nlohmann::json::object();
	cfg.validation = nlohmann::json::object();
	cfg.attach_type = 9; // kretprobe
	return cfg;
}

static std::pair<std::string, bool>
patch_retprobe(const std::string &ptx, const std::string &kernel,
		const std::vector<uint64_t> &ebpf_words, bool warp_auto_execution)
{
	std::string fname = std::string("__retprobe_func__") + kernel;

	auto func_ptx = ptxpass::compile_ebpf_to_ptx_from_words(
		ebpf_words, "sm_61", fname, false, false);
	auto body = ptxpass::find_kernel_body(ptx, kernel);
	if (body.first == std::string::npos) {
		return { ptx, false };
	}
	std::string out = ptx;
	bool warp_lowered = false;
	if (warp_auto_execution) {
		if (!ptxpass::ptx_supports_warp_sync(ptx)) {
			std::fprintf(stderr,
				     "[ptxpass] kretprobe: PTX version lacks vote.ballot.sync; keeping per-thread hook call for %s\n",
				     kernel.c_str());
		} else if (!ptxpass::insert_warp_execution_register_decls(
				   out, kernel)) {
			std::fprintf(stderr,
				     "[ptxpass] kretprobe: cannot insert warp-exec register declarations for %s; keeping per-thread hook call\n",
				     kernel.c_str());
		} else {
			warp_lowered = true;
		}
	}
	// Positions may have shifted after the declaration insertion.
	if (warp_lowered)
	{
		body = ptxpass::find_kernel_body(out, kernel);
		if (body.first == std::string::npos) {
			std::fprintf(stderr,
				     "[ptxpass] kretprobe: lost kernel body for %s after warp-exec insertion\n",
				     kernel.c_str());
			return { ptx, false };
		}
	}
	std::string section = out.substr(body.first, body.second - body.first);
	// PTX kernels can terminate with either 'ret;' or 'exit;'. Some
	// variants are predicated, e.g. '@%p1 exit;'. Keep the predicate (if
	// any) so the injected call matches the original control-flow.
	// Also handle labels on the same line, e.g. '$L0: ret;'. In that case,
	// keep the label on the injected call so branches still land at the
	// epilogue.
	static std::regex retpat(
		R"((\s*)((?:[\w$\.]+:\s*)?)(@[^\s]+\s+)?((?:ret|exit);))");
	if (warp_lowered) {
		// Automatic warp-execution lowering: elect the leader among the
		// lanes executing this site, then let only the leader call the
		// handler once. The original (possibly predicated) ret/exit
		// line is preserved afterwards.
		// std::regex_replace has no callback form. Walk each ret/exit
		// site with sregex_iterator and splice the election preamble
		// in ahead of the preserved (possibly predicated) control-flow
		// line.
		std::string replaced;
		replaced.reserve(section.size() + 1024);
		auto begin = std::sregex_iterator(section.begin(),
						  section.end(), retpat);
		auto end_it = std::sregex_iterator();
		size_t last_end = 0;
		for (auto it = begin; it != end_it; ++it) {
			const auto &m = *it;
			const size_t match_begin =
				(size_t)(m.position() + m[1].length() +
					 m[2].length());
			replaced.append(section, last_end,
					match_begin - last_end);
			// The copied pre-text ($1 + $2, including a same-line
			// label) stays verbatim; terminate that label line,
			// then emit the election preamble and the preserved
			// predicated control-flow line once.
			if (!replaced.empty() && replaced.back() != '\n')
				replaced.push_back('\n');
			const std::string prefix =
				ptxpass::emit_warp_leader_hook_prefix(
					fname, m[3].str());
			replaced.append(prefix);
			replaced.append(m[1].str() + m[3].str() + m[4].str());
			last_end = match_begin + (m[3].length() +
						  m[4].length());
		}
		replaced.append(section, last_end, std::string::npos);
		section = std::move(replaced);
		ptxpass::log_transform_stats(
			"kretprobe-warp", 1, ptx.size(), out.size());
	} else {
		section = std::regex_replace(
			section, retpat,
			std::string("$1$2$3call ") + fname + ";\n$1$3$4");
	}
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
			runtime_request.get_uint64_ebpf_instructions(),
			runtime_request.input.warp_auto_execution);
		snprintf(
			output, length, "%s",
			emit_runtime_response_and_return(out, modified).c_str());
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
