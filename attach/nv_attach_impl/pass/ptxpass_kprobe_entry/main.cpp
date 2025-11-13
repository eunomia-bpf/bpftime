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
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(EntryParams, save_strategy,
						emit_nops_for_alignment,
						pad_nops);

} // namespace entry_params

static ptxpass::pass_config::PassConfig get_default_config()
{
	ptxpass::pass_config::PassConfig cfg;
	cfg.name = "kprobe_entry";
	cfg.description =
		"Instrument PTX at kprobe entry points, excluding __memcapture";
	cfg.attach_points.includes = { "^kprobe/.*$" };
	cfg.attach_points.excludes = { "^kprobe/__memcapture$" };
	cfg.parameters = nlohmann::json{ { "insert_globaltimer", true } };
	cfg.attach_type = 8; // kprobe
	return cfg;
}

static std::pair<std::string, bool>
patch_entry(const std::string &ptx, const std::string &kernel,
	    const std::vector<uint64_t> &ebpf_words)
{
	if (ebpf_words.empty()) {
		return { ptx, false };
	}
	std::string fname = std::string("__probe_func__") + kernel;
	auto func_ptx = ptxpass::compile_ebpf_to_ptx_from_words(
		ebpf_words, "sm_60", fname, true, false);
	auto body = ptxpass::find_kernel_body(ptx, kernel);
	if (body.first == std::string::npos)
		return { ptx, false };
	std::string out = ptx;
	size_t brace = out.find('{', body.first);
	if (brace == std::string::npos)
		return { ptx, false };
	size_t insertPos = brace + 1;
	if (insertPos < out.size() && out[insertPos] == '\n')
		insertPos++;

	out.insert(insertPos, std::string("\n    call ") + fname + ";\n");
	// Insert generated function AFTER PTX headers (before first
	// .entry/.func)
	{
		// Recompile with headers filtered but WITHOUT register guard to
		// reduce risk of illegal accesses
		func_ptx = ptxpass::compile_ebpf_to_ptx_from_words(
			ebpf_words, "sm_60", fname,
			/*add_register_guard*/ false, false);
		size_t insert_pos = std::string::npos;
		auto update_pos = [&](size_t cand) {
			if (cand != std::string::npos) {
				if (insert_pos == std::string::npos ||
				    cand < insert_pos) {
					insert_pos = cand;
				}
			}
		};
		// Beginning of file checks
		if (out.rfind(".visible .entry", 0) == 0)
			update_pos(0);
		if (out.rfind(".entry", 0) == 0)
			update_pos(0);
		if (out.rfind(".visible .func", 0) == 0)
			update_pos(0);
		if (out.rfind(".func", 0) == 0)
			update_pos(0);
		// After newline checks
		update_pos(out.find("\n.visible .entry"));
		update_pos(out.find("\n.entry"));
		update_pos(out.find("\n.visible .func"));
		update_pos(out.find("\n.func"));
		if (insert_pos == std::string::npos)
			insert_pos = out.size();
		out.insert(insert_pos, func_ptx + "\n");
	}
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
		auto runtime_request = pass_runtime_request_from_string(input);
		if (!validate_input(runtime_request.input.full_ptx,
				    cfg.validation)) {
			return ExitCode::TransformFailed;
		}
		auto [out, modified] = patch_entry(
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
