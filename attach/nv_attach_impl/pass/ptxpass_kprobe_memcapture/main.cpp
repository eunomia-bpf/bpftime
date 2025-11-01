#include "json.hpp"
#include "ptxpass/core.hpp"
#include <exception>
#include <iostream>
#include <string>
#include <vector>
#include <regex>
#include <sstream>
#include <ebpf_inst.h>
#include <fstream>
namespace memcapture_params
{
struct MemcaptureParams {
	int buffer_bytes = 4096;
	int max_segments = 4;
	bool allow_partial = true;
	bool emit_nops_for_alignment = false;
	int pad_nops = 0;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(MemcaptureParams, buffer_bytes,
						max_segments, allow_partial,
						emit_nops_for_alignment,
						pad_nops)

} // namespace memcapture_params

static ptxpass::pass_config::PassConfig get_default_config()
{
	ptxpass::pass_config::PassConfig cfg;
	cfg.name = "kprobe_memcapture";
	cfg.description =
		"Emit a probe per ld/st and push source line text to eBPF stack";
	cfg.attach_points.includes = { "^kprobe/__memcapture$" };
	cfg.attach_points.excludes = {};
	cfg.parameters = nlohmann::json::object();
	cfg.validation = nlohmann::json::object();
	cfg.attach_type = 8; // kprobe
	return cfg;
}

static std::pair<std::string, bool>
patch_memcapture(const std::string &ptx,
		 const std::vector<uint64_t> &ebpf_words)
{
	static std::regex ld_st_pattern(
		R"(^\s*(ld|st)\.(const|global|local|param)?\.(((s|u|b)(8|16|32|64))|\.b128|(\.f(16|16x2|32|64))) +(.+), *(.+);\s*$)");
	// no local ebpf_inst dependency here; we will pack words when needed

	std::istringstream iss(ptx);
	std::ostringstream out_body;
	std::ostringstream out_funcs;
	std::string line;
	int count = 0;
	while (std::getline(iss, line)) {
		out_body << line << '\n';
		if (std::regex_match(line, ld_st_pattern)) {
			count++;
			const ebpf_inst *base_insts =
				reinterpret_cast<const ebpf_inst *>(
					ebpf_words.data());
			std::vector<ebpf_inst> insts_local(
				base_insts, base_insts + ebpf_words.size());
			{
				std::vector<ebpf_inst> prep;
				int32_t total_length =
					((int32_t)line.size() + 1 + 7) / 8 * 8;
				{
					ebpf_inst t{};
					t.opcode = EBPF_OP_STDW;
					t.dst = 10;
					t.src = 0;
					t.offset = -8;
					t.imm = 0;
					prep.push_back(t);
				}
				{
					ebpf_inst t{};
					t.opcode = EBPF_OP_SUB64_IMM;
					t.dst = 10;
					t.src = 0;
					t.offset = 0;
					t.imm = total_length;
					prep.push_back(t);
				}
				{
					ebpf_inst t{};
					t.opcode = EBPF_OP_MOV64_REG;
					t.dst = 1;
					t.src = 10;
					t.offset = 0;
					t.imm = 0;
					prep.push_back(t);
				}
				for (int i = 0; i < (int)line.size(); i += 8) {
					uint64_t curr = 0;
					for (int j = 0;
					     j < 8 &&
					     (j + i) < (int)line.size();
					     j++)
						curr |= ((uint64_t)line[i + j])
							<< (j * 8);
					{
						ebpf_inst t{};
						t.opcode = EBPF_OP_LDDW;
						t.dst = 2;
						t.src = 0;
						t.offset = 0;
						t.imm = (int32_t)(uint32_t)curr;
						prep.push_back(t);
					}
					{
						ebpf_inst t{};
						t.opcode = 0;
						t.dst = 0;
						t.src = 0;
						t.offset = 0;
						t.imm = (int32_t)(uint32_t)(curr >>
									    32);
						prep.push_back(t);
					}
					{
						ebpf_inst t{};
						t.opcode = EBPF_OP_STXDW;
						t.dst = 1;
						t.src = 2;
						t.offset = (int16_t)i;
						t.imm = 0;
						prep.push_back(t);
					}
				}
				insts_local.insert(insts_local.begin(),
						   prep.begin(), prep.end());
			}
			const uint64_t *packed_words =
				reinterpret_cast<const uint64_t *>(
					insts_local.data());
			std::vector<uint64_t> packed(
				packed_words,
				packed_words + insts_local.size());
			auto func_ptx = ptxpass::compile_ebpf_to_ptx_from_words(
				packed, "sm_60",
				"__memcapture__" + std::to_string(count), true,
				false);
			auto func_name = std::string("__memcapture__") +
					 std::to_string(count);
			out_funcs << func_ptx << "\n";
			out_body << "call " << func_name << ";\n";
		}
	}
	if (count == 0)
		return { "", false };
	auto out = out_funcs.str() + "\n" + out_body.str();
	ptxpass::log_transform_stats("kprobe_memcapture", count, ptx.size(),
				     out.size());
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
			std::ifstream ifs(config_path);
			auto input_json = nlohmann::json::parse(ifs);
			pass_config::from_json(input_json, cfg);
		} else {
			cfg = get_default_config();
		}
		auto matcher = AttachPointMatcher(cfg.attach_points);

		std::string stdin_data = read_all_from_stdin();
		auto runtime_request =
			pass_runtime_request_from_string(stdin_data);

		if (dry_run) {
			emit_runtime_response_and_print("");
			return ExitCode::Success;
		}
		if (!validate_input(runtime_request.input.full_ptx,
				    cfg.validation)) {
			return ExitCode::TransformFailed;
		}
		auto [out, modified] =
			patch_memcapture(runtime_request.input.full_ptx,
					 runtime_request.ebpf_instructions);
		emit_runtime_response_and_print(modified ? out : "");
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
