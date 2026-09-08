#include "json.hpp"
#include "ptxpass/core.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <exception>
#include <iostream>
#include <string>
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
	    const std::string &stub_name, bool add_register_guard,
	    bool warp_auto_execution)
{
	if (ebpf_words.empty()) {
		return { ptx, false };
	}
	std::string fname = std::string("__probe_func__") + kernel;
	auto func_ptx = ptxpass::compile_ebpf_to_ptx_from_words(
		ebpf_words, "sm_61", fname, add_register_guard, false);
	std::string out = ptx;

	bool patched_stub_calls = false;
	bool warp_decls_inserted = false;
	{
		// Warp lowering is scoped to the selected kernel body: the
		// leader registers are only declared there, so stub calls in
		// other functions must keep their original form.
		auto stub_body = warp_auto_execution ?
					 ptxpass::find_kernel_body(out, kernel) :
					 std::pair<size_t, size_t>{
						 std::string::npos,
						 std::string::npos };
		bool warp_stub_lowered =
			stub_body.first != std::string::npos &&
			ptxpass::ptx_supports_warp_sync(out) &&
			(warp_decls_inserted = ptxpass::
				insert_warp_execution_register_decls(out,
								     kernel));
		const std::string patterns[] = { "call " + stub_name,
						 "call.uni " + stub_name };
		for (const auto &pat : patterns) {
			size_t search_begin = 0;
			size_t body_end = std::string::npos;
			if (warp_stub_lowered) {
				// Re-find the body: earlier pattern passes
				// shifted positions.
				stub_body = ptxpass::find_kernel_body(out,
								      kernel);
				if (stub_body.first == std::string::npos)
					break;
				search_begin = stub_body.first;
				body_end = stub_body.second;
			}
			size_t pos = search_begin;
			size_t end_shift = 0;
			while (true) {
				pos = out.find(pat, pos);
				if (pos == std::string::npos)
					break;
				if (body_end != std::string::npos &&
				    pos + pat.size() > body_end + end_shift)
					break;
				std::string replacement;
				size_t replace_begin;
				size_t replace_len;
				if (warp_stub_lowered) {
					// Whole-statement rewrite: the site
					// predicate is consumed into the
					// ballot, an existing label on the
					// line is preserved once, and the
					// terminator/operands after the
					// callee stay in place to terminate
					// the emitted leader call.
					const size_t line_start =
						out.rfind('\n', pos);
					const size_t stmt_begin =
						line_start ==
								std::string::npos ?
							0 :
							line_start + 1;
					const size_t call_end =
						pos + pat.size();
					const std::string head =
						out.substr(stmt_begin,
							   pos - stmt_begin);
					std::smatch pm;
					static const std::regex headpat(
						R"(^([ \t]*)((?:[\w$\.]+:[ \t]*)?)(@[^\s]+[ \t]*)?$)");
					if (std::regex_match(head, pm,
							     headpat)) {
						std::string prefix =
							ptxpass::
								emit_warp_leader_hook_prefix(
									fname,
									pm[3].str());
						// The helper emits the guarded
						// call with a trailing ";\n";
						// splice it off so the
						// original statement
						// terminator remains.
						if (prefix.size() >= 2 &&
						    prefix.compare(
							    prefix.size() - 2,
							    2,
							    ";\n") == 0)
							prefix.resize(
								prefix.size() -
								2);
						replacement = pm[1].str() +
							      pm[2].str();
						if (!pm[2].str().empty())
							replacement += "\n";
						replacement += prefix;
						replace_begin = stmt_begin;
					} else {
						// Unexpected tokens precede
						// the call on this line; keep
						// them and rewrite only the
						// callee, as in the non-warp
						// path.
						replacement =
							pat.substr(
								0,
								pat.find(' ')) +
							" " + fname;
						replace_begin = pos;
					}
					replace_len = call_end - replace_begin;
				} else {
					replacement = pat.substr(
							      0,
							      pat.find(' ')) +
						      " " + fname;
					replace_begin = pos;
					replace_len = pat.size();
				}
				out.replace(replace_begin, replace_len,
					    replacement);
				end_shift +=
					replacement.size() - replace_len;
				pos = replace_begin + replacement.size();
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
	bool warp_entry_lowered =
		warp_auto_execution && ptxpass::ptx_supports_warp_sync(out) &&
		(warp_decls_inserted ||
		 ptxpass::insert_warp_execution_register_decls(out, kernel));
	if (warp_entry_lowered)
		// Declaration insertion shifted offsets; recompute the body
		// range on the modified PTX.
		body = ptxpass::find_kernel_body(out, kernel);
	size_t brace = out.find('{', body.first);
	if (brace == std::string::npos)
		return { ptx, false };
	size_t insertPos = brace + 1;
	if (insertPos < out.size() && out[insertPos] == '\n')
		insertPos++;

	if (warp_entry_lowered) {
		// Kernel-entry hook: every launched lane of the warp is still
		// active and converged here, so the ballot elects the leader
		// once per active warp.
		out.insert(insertPos,
			   "\n" + ptxpass::emit_warp_leader_hook_prefix(fname, "") +
				   "\n");
	} else {
		out.insert(insertPos, std::string("\n    call ") + fname + ";\n");
	}
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

extern "C" int process_input(const char *input, int length, char *output)
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
		bool add_register_guard = params.save_strategy == "full";

		auto runtime_request = pass_runtime_request_from_string(input);
		if (!validate_input(runtime_request.input.full_ptx,
				    cfg.validation)) {
			return ExitCode::TransformFailed;
		}
		auto [out, modified] = patch_entry(
			runtime_request.input.full_ptx,
			runtime_request.input.to_patch_kernel,
			runtime_request.get_uint64_ebpf_instructions(),
			stub_name, add_register_guard,
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
