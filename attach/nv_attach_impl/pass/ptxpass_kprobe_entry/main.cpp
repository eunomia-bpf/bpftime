#include "json.hpp"
#include "ptxpass/core.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <exception>
#include <iostream>
#include <string>
#include <string_view>
namespace entry_params
{
struct EntryParams {
	std::string save_strategy = "minimal"; // "minimal" or "full"
	bool emit_nops_for_alignment = false;
	int pad_nops = 0;
	std::string stub_name;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(EntryParams, save_strategy,
						emit_nops_for_alignment,
						pad_nops, stub_name);

} // namespace entry_params

static ptxpass::pass_config::PassConfig get_default_config()
{
	ptxpass::pass_config::PassConfig cfg;
	cfg.name = "kprobe_entry";
	cfg.description =
		"Instrument PTX at kprobe entry points, excluding __memcapture";
	cfg.attach_points.includes = { "^kprobe/.*$" };
	cfg.attach_points.excludes = { "^kprobe/__memcapture$" };
	// Parameters:
	// - insert_globaltimer: legacy flag to control timestamp injection
	// - stub_name: CUDA device stub used as hook point; calls to this
	//   function will be rewritten to the eBPF-generated probe PTX.
	entry_params::EntryParams params;
	params.save_strategy = "minimal";
	params.emit_nops_for_alignment = false;
	params.pad_nops = 0;
	params.stub_name = "__bpftime_cuda__kernel_trace";
	cfg.parameters = params;
	cfg.attach_type = 8; // kprobe
	return cfg;
}

static std::pair<std::string, bool>
patch_entry(const std::string &ptx, const std::string &kernel,
	    const std::vector<uint64_t> &ebpf_words,
	    const std::string &stub_name, bool add_register_guard)
{
	if (ebpf_words.empty()) {
		return { ptx, false };
	}
	std::string fname = std::string("__probe_func__") + kernel;
	auto func_ptx = ptxpass::compile_ebpf_to_ptx_from_words(
		ebpf_words, "sm_61", fname, add_register_guard, false);
	std::string out = ptx;

	bool patched_stub_calls = false;
	{
		const std::string patterns[] = { "call " + stub_name,
						 "call.uni " + stub_name };
		for (const auto &pat : patterns) {
			size_t pos = 0;
			while ((pos = out.find(pat, pos)) !=
			       std::string::npos) {
				std::string replacement =
					pat.substr(0, pat.find(' ')) + " " +
					fname;
				out.replace(pos, pat.size(), replacement);
				pos += replacement.size();
				patched_stub_calls = true;
			}
		}
	}

	if (patched_stub_calls) {
		out = func_ptx + "\n" + out;
		ptxpass::log_transform_stats("kprobe_entry_stub", 1,
					     ptx.size(), out.size());
		return { out, true };
	}

	auto body = ptxpass::find_kernel_body(ptx, kernel);
	if (body.first == std::string::npos)
		return { ptx, false };
	size_t brace = out.find('{', body.first);
	if (brace == std::string::npos)
		return { ptx, false };
	size_t insertPos = brace + 1;
	if (insertPos < out.size() && out[insertPos] == '\n')
		insertPos++;

	out.insert(insertPos, std::string("\n    call ") + fname + ";\n");
	out = func_ptx + "\n" + out;
	ptxpass::log_transform_stats("kprobe_entry", 1, ptx.size(), out.size());
	return { out, true };
}

extern "C" void print_config(int length, char *out)
{
	auto cfg = get_default_config();
	nlohmann::json output_json;
	ptxpass::pass_config::to_json(output_json, cfg);
	snprintf(out, length, "%s", output_json.dump().c_str());
}

extern "C" int process_input(const char *ptx_text, size_t ptx_len,
			     const char *meta_json, int meta_len,
			     char *output, int output_len)
{
	using namespace ptxpass;
	auto cfg = get_default_config();
	try {
		// Decode parameters into a typed struct so extended fields
		// (like stub_name) remain type-safe.
		entry_params::EntryParams params =
			cfg.parameters.get<entry_params::EntryParams>();
		std::string stub_name = params.stub_name.empty()
			? "__bpftime_cuda__kernel_trace"
			: params.stub_name;
		if (meta_json == nullptr) {
			throw std::runtime_error("Metadata JSON is missing");
		}
		bool add_register_guard = params.save_strategy == "full";

		auto runtime_request = pass_runtime_request_from_string(
			std::string(meta_json, meta_len));
		std::string_view ptx_view = ptx_text != nullptr
			? std::string_view(ptx_text, ptx_len)
			: std::string_view(runtime_request.full_ptx);
		if (ptx_text == nullptr && runtime_request.full_ptx.empty()) {
			throw std::runtime_error("PTX input is missing");
		}
		if (!runtime_request.input.to_patch_kernel.empty() &&
		    ptx_view.find(runtime_request.input.to_patch_kernel) ==
			    std::string_view::npos) {
			snprintf(output, output_len, "%s",
				 emit_runtime_response_and_return("",
								  false)
					 .c_str());
			return ExitCode::Success;
		}
		if (!validate_input(std::string(ptx_view), cfg.validation)) {
			return ExitCode::TransformFailed;
		}
		if (ptx_text != nullptr) {
			runtime_request.full_ptx.assign(ptx_text, ptx_len);
		}
		auto [out, modified] = patch_entry(
			runtime_request.full_ptx,
			runtime_request.input.to_patch_kernel,
			runtime_request.get_uint64_ebpf_instructions(),
			stub_name, add_register_guard);
		snprintf(output, output_len, "%s",
			 emit_runtime_response_and_return(out, modified)
				 .c_str());
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
