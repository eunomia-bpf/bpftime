#include "ptxpass/core.hpp"
#include <exception>
#include <iostream>
#include <string>
#include <vector>
#include <regex>
#include <sstream>
#include <ebpf_inst.h>

static ptxpass::PassConfig GetDefaultConfig()
{
	ptxpass::PassConfig cfg;
	cfg.name = "kprobe_memcapture";
	cfg.description =
		"Emit a probe per ld/st and push source line text to eBPF stack";
	cfg.attachPoints.includes = { "^kprobe/__memcapture$" };
	cfg.attachPoints.excludes = {};
	cfg.parameters = nlohmann::json::object();
	cfg.validation = nlohmann::json::object();
	return cfg;
}

static std::pair<std::string, bool>
PatchMemcapture(const std::string &ptx, const std::vector<uint64_t> &ebpf_words)
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
			// rebuild ebpf_inst list from words each time
			std::vector<ebpf_inst> insts_local;
			insts_local.reserve(ebpf_words.size());
			for (auto w : ebpf_words) {
				ebpf_inst ins{};
				ins.opcode = (uint8_t)(w & 0xFF);
				ins.dst = (uint8_t)((w >> 8) & 0xF);
				ins.src = (uint8_t)((w >> 12) & 0xF);
				ins.offset = (int16_t)((w >> 16) & 0xFFFF);
				ins.imm = (int32_t)(w >> 32);
				insts_local.push_back(ins);
			}
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
			std::vector<uint64_t> packed;
			packed.reserve(insts_local.size());
			for (auto &ins : insts_local) {
				uint64_t w = 0;
				w |= (uint64_t)ins.opcode;
				w |= (uint64_t)ins.dst << 8;
				w |= (uint64_t)ins.src << 12;
				w |= (uint64_t)(uint16_t)ins.offset << 16;
				w |= (uint64_t)(uint32_t)ins.imm << 32;
				packed.push_back(w);
			}
			auto func_ptx = ptxpass::compile_ebpf_to_ptx_from_words(
				packed, "sm_60");
			auto func_name = std::string("__memcapture__") +
					 std::to_string(count);
			out_funcs << ".func " << func_name << "\n"
				  << func_ptx << "\n";
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
		// JSON-only
		if (!isJson) {
			return ExitCode::InputError;
		}
		if (!matcher.matches(ap)) {
			emit_runtime_output("");
			return ExitCode::Success;
		}
		if (dryRun) {
			emit_runtime_output("");
			return ExitCode::Success;
		}
		if (!validate_input(rr.input.full_ptx, cfg.validation)) {
			return ExitCode::TransformFailed;
		}
		auto [out, modified] = PatchMemcapture(rr.input.full_ptx,
						       rr.ebpf_instructions);
		emit_runtime_output(modified ? out : "");
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
