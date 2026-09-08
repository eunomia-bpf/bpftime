#include "catch2/catch_test_macros.hpp"
#include "ptxpass/core.hpp"
#include <memory>
#include <fstream>
#include <iostream>
#include <spdlog/logger.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>
#include <vector>

using namespace ptxpass;

static inline runtime_input::RuntimeInput
pass_runtime_input_from_string(const std::string &str)
{
	using namespace runtime_input;
	RuntimeInput runtime_input;
	auto input_json = nlohmann::json::parse(str);
	from_json(input_json, runtime_input);
	return runtime_input;
}

static const std::string MINIMAL_PTX = R"(.version 7.0
.target sm_60
.address_size 64

.visible .entry test_kernel()
{
    ret;
}
)";

static const std::string KERNEL_WITH_BODY = R"(.version 7.0
.target sm_60
.address_size 64

.visible .entry foo()
{
    mov.u32 %r1, 42;
    ret;
}
)";

TEST_CASE("read_all_from_stdin reads from standard input", "[ptxpass_core]")
{
	SECTION("Empty input is handled")
	{
		std::istringstream empty_input("");
		auto *original_buffer = std::cin.rdbuf(empty_input.rdbuf());
		std::string result = read_all_from_stdin();
		std::cin.rdbuf(original_buffer);
		REQUIRE(result.empty());
	}
}

TEST_CASE("is_whitespace_only detects whitespace-only strings",
	  "[ptxpass_core]")
{
	REQUIRE(is_whitespace_only(""));
	REQUIRE(is_whitespace_only("   "));
	REQUIRE(is_whitespace_only("\n\t\r"));
	REQUIRE(is_whitespace_only("  \n\t  \r\n  "));
	REQUIRE_FALSE(is_whitespace_only("a"));
	REQUIRE_FALSE(is_whitespace_only("  x  "));
	REQUIRE_FALSE(is_whitespace_only("\nabc\n"));
}

TEST_CASE("get_env retrieves environment variables", "[ptxpass_core]")
{
	std::string path = get_env("PATH");
	REQUIRE_FALSE(path.empty());

	std::string nonexistent = get_env("NONEXISTENT_VAR_12345");
	REQUIRE(nonexistent.empty());
}

TEST_CASE("parse_runtime_input parses JSON input", "[ptxpass_core]")
{
	SECTION("Valid JSON with full_ptx and to_patch_kernel")
	{
		std::string json_input = R"({
  "full_ptx": ".version 7.0\n.target sm_60\n",
  "to_patch_kernel": "test_kernel"
})";
		auto input = pass_runtime_input_from_string(json_input);
		REQUIRE(input.full_ptx == ".version 7.0\n.target sm_60\n");
		REQUIRE(input.to_patch_kernel == "test_kernel");
	}

	SECTION("Invalid JSON returns false")
	{
		std::string invalid_json = "not json at all";
		REQUIRE_THROWS(pass_runtime_input_from_string(invalid_json));
	}
}

TEST_CASE("parse_runtime_request parses full request with ebpf_instructions",
	  "[ptxpass_core]")
{
	SECTION("Valid request with ebpf_instructions")
	{
		std::string json_request = R"({
			"input":{
  "full_ptx": ".version 7.0\n",
  "to_patch_kernel": "foo"
},
  "ebpf_instructions": [{"upper_32bit":0,"lower_32bit":100}, {"upper_32bit":0,"lower_32bit":200}, {"upper_32bit":0,"lower_32bit":300}]
		})";
		auto request = pass_runtime_request_from_string(json_request);
		REQUIRE(request.input.full_ptx == ".version 7.0\n");
		REQUIRE(request.input.to_patch_kernel == "foo");
		REQUIRE(request.ebpf_instructions.size() == 3);
		REQUIRE(request.ebpf_instructions[0].to_uint64() == 100);
		REQUIRE(request.ebpf_instructions[1].to_uint64() == 200);
		REQUIRE(request.ebpf_instructions[2].to_uint64() == 300);
	}
}

TEST_CASE("emit_runtime_output produces JSON output", "[ptxpass_core]")
{
	std::ostringstream oss;
	std::streambuf *old_cout = std::cout.rdbuf(oss.rdbuf());

	emit_runtime_response_and_print("test_ptx_output");

	std::cout.rdbuf(old_cout);

	std::string output = oss.str();
	REQUIRE(output.find("\"output_ptx\"") != std::string::npos);
	REQUIRE(output.find("test_ptx_output") != std::string::npos);
}

TEST_CASE("contains_entry_function detects .visible .entry", "[ptxpass_core]")
{
	REQUIRE(contains_entry_function(MINIMAL_PTX));
	REQUIRE(contains_entry_function(".visible .entry foo()"));
	REQUIRE_FALSE(contains_entry_function(".func bar()"));
	REQUIRE_FALSE(contains_entry_function(""));
}

TEST_CASE("contains_ret_instruction detects ret instruction", "[ptxpass_core]")
{
	REQUIRE(contains_ret_instruction(MINIMAL_PTX));
	REQUIRE(contains_ret_instruction("{\n    ret;\n}"));
	REQUIRE(contains_ret_instruction("{\n\tret;\n}"));
	REQUIRE_FALSE(contains_ret_instruction("no return here"));
}

TEST_CASE("validate_ptx_version checks PTX version", "[ptxpass_core]")
{
	std::string ptx_v7 = ".version 7.0\n.target sm_60\n";
	std::string ptx_v6 = ".version 6.5\n.target sm_60\n";

	REQUIRE(validate_ptx_version(ptx_v7, "6.0"));
	REQUIRE(validate_ptx_version(ptx_v7, "7.0"));
	REQUIRE_FALSE(validate_ptx_version(ptx_v6, "7.0"));
	REQUIRE(validate_ptx_version(ptx_v6, "6.0"));
}

TEST_CASE("validate_input checks input against validation rules",
	  "[ptxpass_core]")
{
	nlohmann::json validation_require_entry = { { "require_entry", true } };
	nlohmann::json validation_require_ret = { { "require_ret", true } };
	nlohmann::json validation_version_min = { { "ptx_version_min",
						    "7.0" } };

	REQUIRE(validate_input(MINIMAL_PTX, validation_require_entry));
	REQUIRE(validate_input(MINIMAL_PTX, validation_require_ret));
	REQUIRE(validate_input(MINIMAL_PTX, validation_version_min));

	std::string no_entry = ".version 7.0\n.func bar() { ret; }\n";
	REQUIRE_FALSE(validate_input(no_entry, validation_require_entry));

	std::string no_ret =
		".version 7.0\n.visible .entry foo() { bra LOOP; }";
	REQUIRE_FALSE(validate_input(no_ret, validation_require_ret));

	std::string old_version =
		".version 6.0\n.visible .entry foo() { ret; }";
	REQUIRE_FALSE(validate_input(old_version, validation_version_min));
}

TEST_CASE("filter_out_version_headers_ptx removes duplicate header lines",
	  "[ptxpass_core]")
{
	std::string input = R"(.version 7.0
.target sm_60
.address_size 64
// comment 1

.visible .entry foo()
{
    ret;
}

.version 7.0
.target sm_60
// comment 2
)";

	std::string filtered = filter_out_version_headers_ptx(input);
	size_t first_version_pos = filtered.find(".version");
	REQUIRE(first_version_pos != std::string::npos);
	size_t second_version_pos =
		filtered.find(".version", first_version_pos + 1);
	REQUIRE(second_version_pos == std::string::npos);

	size_t first_target_pos = filtered.find(".target");
	REQUIRE(first_target_pos != std::string::npos);
	size_t second_target_pos =
		filtered.find(".target", first_target_pos + 1);
	REQUIRE(second_target_pos == std::string::npos);

	REQUIRE(filtered.find(".visible .entry foo()") != std::string::npos);
	REQUIRE(filtered.find("ret;") != std::string::npos);
}

TEST_CASE("find_kernel_body locates kernel boundaries", "[ptxpass_core]")
{
	auto [start, end] = find_kernel_body(KERNEL_WITH_BODY, "foo");
	REQUIRE(start != std::string::npos);
	REQUIRE(end != std::string::npos);
	REQUIRE(start < end);

	std::string section = KERNEL_WITH_BODY.substr(start, end - start);
	REQUIRE(section.find(".visible .entry foo()") != std::string::npos);
	REQUIRE(section.find("mov.u32") != std::string::npos);
	REQUIRE(section.find("ret;") != std::string::npos);

	auto [not_found_start, not_found_end] =
		find_kernel_body(KERNEL_WITH_BODY, "nonexistent");
	REQUIRE(not_found_start == std::string::npos);
	REQUIRE(not_found_end == std::string::npos);
}

TEST_CASE("AttachPointMatcher matches attach points", "[ptxpass_core]")
{
	attach_points::AttachPoints points;
	points.includes = { "kprobe/.*", "kretprobe/.*" };
	points.excludes = { "kprobe/do_not_match" };

	AttachPointMatcher matcher(points);

	REQUIRE(matcher.matches("kprobe/foo"));
	REQUIRE(matcher.matches("kprobe/bar_123"));
	REQUIRE(matcher.matches("kretprobe/baz"));
	REQUIRE_FALSE(matcher.matches("kprobe/do_not_match"));
	REQUIRE_FALSE(matcher.matches("uprobe/something"));
}

TEST_CASE("JsonConfigLoader loads config from file", "[ptxpass_core]")
{
	std::string temp_config_path = "/tmp/test_ptxpass_config.json";
	std::ofstream ofs(temp_config_path);
	ofs << R"({
  "attach_points": {
    "includes": ["kprobe/.*"],
    "excludes": []
  },
  "parameters": {
    "save_strategy": "minimal"
  },
  "validation": {
    "require_entry": true
  },
  "name":"test",
  "description":"test",
  "attach_type":1001
})";
	ofs.close();

	pass_config::PassConfig cfg =
		load_pass_config_from_file(temp_config_path);
	REQUIRE(cfg.attach_points.includes.size() == 1);
	REQUIRE(cfg.attach_points.includes[0] == "kprobe/.*");
	REQUIRE(cfg.attach_points.excludes.empty());
	REQUIRE(cfg.validation["require_entry"].get<bool>());

	std::remove(temp_config_path.c_str());
}

TEST_CASE("compile_ebpf_to_ptx_from_words compiles eBPF to PTX",
	  "[ptxpass_core]")
{
	std::vector<uint64_t> nop_ebpf = { 0x0000000000000095ULL };

	std::string ptx = compile_ebpf_to_ptx_from_words(
		nop_ebpf, "sm_60", "__probe__", true, false);

	REQUIRE_FALSE(ptx.empty());
	REQUIRE(ptx.find("ret") != std::string::npos);
}

TEST_CASE("log_transform_stats bypasses the application default logger",
	  "[ptxpass_core]")
{
	std::ostringstream oss;
	auto old_logger = spdlog::default_logger();
	auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(oss);
	auto logger = std::make_shared<spdlog::logger>("ptxpass_test", sink);
	logger->set_level(spdlog::level::info);
	spdlog::set_default_logger(logger);

	log_transform_stats("test_pass", 5, 1024, 2048);
	logger->flush();
	spdlog::set_default_logger(old_logger);

	// The process-level plugin test separately checks the real stderr bytes.
	REQUIRE(oss.str().empty());
}

TEST_CASE("PassConfig default includes and excludes work correctly",
	  "[ptxpass_core][integration]")
{
	SECTION("kprobe_entry default config")
	{
		pass_config::PassConfig cfg;
		cfg.attach_points.includes = { "^kprobe/.*$" };
		cfg.attach_points.excludes = { "^kprobe/__memcapture$" };

		AttachPointMatcher matcher(cfg.attach_points);

		REQUIRE(matcher.matches("kprobe/test"));
		REQUIRE(matcher.matches("kprobe/sys_read"));
		REQUIRE_FALSE(matcher.matches("kprobe/__memcapture"));
		REQUIRE_FALSE(matcher.matches("kretprobe/test"));
	}

	SECTION("kretprobe default config")
	{
		pass_config::PassConfig cfg;
		cfg.attach_points.includes = { "^kretprobe/.*$" };

		AttachPointMatcher matcher(cfg.attach_points);

		REQUIRE(matcher.matches("kretprobe/test"));
		REQUIRE(matcher.matches("kretprobe/sys_read"));
		REQUIRE_FALSE(matcher.matches("kprobe/test"));
	}

	SECTION("memcapture default config")
	{
		pass_config::PassConfig cfg;
		cfg.attach_points.includes = { "^kprobe/__memcapture$" };

		AttachPointMatcher matcher(cfg.attach_points);

		REQUIRE(matcher.matches("kprobe/__memcapture"));
		REQUIRE_FALSE(matcher.matches("kprobe/test"));
		REQUIRE_FALSE(matcher.matches("kretprobe/__memcapture"));
	}
}

TEST_CASE("RuntimeInput warp_auto_execution round-trips through JSON",
	  "[ptxpass_core]")
{
	SECTION("present true")
	{
		auto ri = pass_runtime_input_from_string(
			R"({"full_ptx":"x","to_patch_kernel":"k",)"
			R"("warp_auto_execution":true})");
		REQUIRE(ri.warp_auto_execution);
	}
	SECTION("present false")
	{
		auto ri = pass_runtime_input_from_string(
			R"({"full_ptx":"x","to_patch_kernel":"k",)"
			R"("warp_auto_execution":false})");
		REQUIRE_FALSE(ri.warp_auto_execution);
	}
	SECTION("absent defaults to false")
	{
		auto ri = pass_runtime_input_from_string(
			R"({"full_ptx":"x","to_patch_kernel":"k"})");
		REQUIRE_FALSE(ri.warp_auto_execution);
	}
	SECTION("serialization include warp_auto_execution")
	{
		runtime_input::RuntimeInput ri;
		ri.warp_auto_execution = true;
		nlohmann::json j;
		nlohmann::to_json(j, ri);
		REQUIRE(j.value("warp_auto_execution", false));
	}
}

TEST_CASE("ptx_supports_warp_sync checks the PTX ISA version",
	  "[ptxpass_core]")
{
	auto ptx_with_version = [](const std::string &version) {
		return ".version " + version +
		       "\n.target sm_60\n"
		       ".visible .entry k() {\n    ret;\n}\n";
	};
	REQUIRE(ptx_supports_warp_sync(ptx_with_version("6.0")));
	REQUIRE(ptx_supports_warp_sync(ptx_with_version("7.0")));
	REQUIRE(ptx_supports_warp_sync(ptx_with_version("8.3")));
	REQUIRE_FALSE(ptx_supports_warp_sync(ptx_with_version("5.0")));
	REQUIRE_FALSE(ptx_supports_warp_sync(
		".target sm_60\n.visible .entry k() {\n    ret;\n}\n"));
}

TEST_CASE("warp execution register declarations are inserted once per kernel",
	  "[ptxpass_core]")
{
	const std::string ptx = ".version 7.0\n.target sm_60\n"
				".visible .entry picked_kernel() {\n"
				"    mov.u32 %r1, 0;\n"
				"    ret;\n"
				"}\n"
				".visible .entry other_kernel() {\n"
				"    ret;\n"
				"}\n";
	std::string out = ptx;
	REQUIRE(ptxpass::insert_warp_execution_register_decls(out,
							      "picked_kernel"));
	// Registers land inside the selected kernel body only.
	REQUIRE(out.find("%r_bpftime_warp0") != std::string::npos);
	const size_t body_start =
		out.find(".visible .entry picked_kernel()");
	REQUIRE(body_start != std::string::npos);
	const size_t body_end = out.find(".visible .entry other_kernel()");
	REQUIRE(body_end != std::string::npos);
	REQUIRE(out.find("%r_bpftime_warp0", body_start) < body_end);
	REQUIRE(out.find("%r_bpftime_warp0", body_end) ==
		std::string::npos);
	// Idempotent: a second insertion must not duplicate declarations.
	REQUIRE_FALSE(ptxpass::insert_warp_execution_register_decls(
		out, "picked_kernel"));
	REQUIRE(out.find("%r_bpftime_warp0", body_end) ==
		std::string::npos);
	const size_t first =
		out.find(".reg .b32 %r_bpftime_warp0");
	REQUIRE(first != std::string::npos);
	REQUIRE(out.find(".reg .b32 %r_bpftime_warp0", first + 1) ==
		std::string::npos);
}

TEST_CASE("emit_warp_leader_hook_prefix emits leader election preamble",
	  "[ptxpass_core]")
{
	SECTION("unconditional site")
	{
		auto prefix =
			emit_warp_leader_hook_prefix("__probe_func__k", "");
		REQUIRE(prefix.find("mov.u32 %r_bpftime_warp0, %laneid;") !=
			std::string::npos);
		REQUIRE(prefix.find("activemask.b32 %r_bpftime_warp3;") !=
			std::string::npos);
		REQUIRE(prefix.find("vote.ballot.sync") == std::string::npos);
		REQUIRE(prefix.find("setp.eq.u32 %p_bpftime_warp0, "
				    "%r_bpftime_warp1, 0;") !=
			std::string::npos);
		REQUIRE(prefix.find("@%p_bpftime_warp0 "
				    "call __probe_func__k;") !=
			std::string::npos);
		REQUIRE(prefix.compare(prefix.size() - 2, 2, ";\n") == 0);
	}
	SECTION("predicated site consumes the predicate")
	{
		auto prefix = emit_warp_leader_hook_prefix("__probe_func__k",
							   "@%p1 ");
		REQUIRE(prefix.find("vote.ballot.sync.b32 "
				    "%r_bpftime_warp3, %p1, "
				    "%r_bpftime_warp2;") !=
			std::string::npos);
		REQUIRE(prefix.find("and.pred %p_bpftime_warp2, "
				    "%p_bpftime_warp0, %p1;") !=
			std::string::npos);
		REQUIRE(prefix.find("@%p_bpftime_warp2 "
				    "call __probe_func__k;") !=
			std::string::npos);
	}
	SECTION("negated site keeps the negation")
	{
		auto prefix = emit_warp_leader_hook_prefix("__probe_func__k",
							   "@!%p2 ");
		REQUIRE(prefix.find("vote.ballot.sync.b32 "
				    "%r_bpftime_warp3, !%p2, "
				    "%r_bpftime_warp2;") !=
			std::string::npos);
		REQUIRE(prefix.find("not.pred %p_bpftime_warp1, %p2;") !=
			std::string::npos);
		REQUIRE(prefix.find("@%p_bpftime_warp2 "
				    "call __probe_func__k;") !=
			std::string::npos);
	}
}

// The kprobe_entry pass is built without its own main(); its sources are
// compiled into this test binary so the transform can be exercised
// end-to-end through the C ABI.
extern "C" int process_input(const char *input, int length, char *output);

static inline nlohmann::json
run_kprobe_entry_pass(const std::string &ptx,
		      const nlohmann::json &input_extra,
		      bool include_warp_flag = true)
{
	nlohmann::json input = { { "full_ptx", ptx },
				 { "to_patch_kernel", "test_kernel" } };
	if (include_warp_flag)
		input.update(input_extra);
	nlohmann::json request = {
		{ "input", input },
		{ "ebpf_instructions",
		  nlohmann::json::array(
			  { { { "upper_32bit", 0 },
			      { "lower_32bit", 0x95 } } }) }
	};
	auto serialized = request.dump();
	std::vector<char> buf(10 << 20);
	int rc = process_input(serialized.data(), (int)buf.size(),
			       buf.data());
	REQUIRE(rc == 0);
	return nlohmann::json::parse(std::string(buf.data()));
}

static inline std::string count_substrings(const std::string &s,
					   const std::string &needle)
{
	std::string::size_type pos = 0;
	size_t count = 0;
	while ((pos = s.find(needle, pos)) != std::string::npos) {
		count++;
		pos += needle.size();
	}
	return std::to_string(count);
}

TEST_CASE("kprobe_entry warp lowering rewrites stub call statements",
	  "[ptxpass][kprobe_entry]")
{
	const std::string ptx = R"(.version 7.0
.target sm_61
.address_size 64

.visible .entry test_kernel()
{
    .reg .pred %p1;
    mov.u32 %r1, %tid.x;
    call __bpftime_cuda__kernel_trace;
    @%p1 call __bpftime_cuda__kernel_trace;
    $L_1: @%p1 call __bpftime_cuda__kernel_trace;
    ret;
}

.visible .entry other_kernel()
{
    call __bpftime_cuda__kernel_trace;
    ret;
}
)";

	auto response = run_kprobe_entry_pass(
		ptx, nlohmann::json{ { "warp_auto_execution", true } });
	REQUIRE(response.value("modified", false));
	const std::string out =
		response.value("output_ptx", std::string());

	// Three stub sites in the selected kernel are lowered.
	REQUIRE(count_substrings(out,
				 "@%p_bpftime_warp0 call __probe_func__"
				 "test_kernel;") == "1");
	// Predicated sites consume the predicate into the ballot and guard
	// the call with the combined predicate.
	REQUIRE(count_substrings(out,
				 "%r_bpftime_warp3, %p1, "
				 "%r_bpftime_warp2;") == "2");
	REQUIRE(count_substrings(out,
				 "@%p_bpftime_warp2 call __probe_func__"
				 "test_kernel;") == "2");
	// The label is preserved exactly once.
	REQUIRE(count_substrings(out, "$L_1:") == "1");
	// Registers declared once, in the selected kernel only.
	REQUIRE(count_substrings(out, ".reg .b32 %r_bpftime_warp0") == "1");
	const size_t other = out.find(".entry other_kernel");
	REQUIRE(other != std::string::npos);
	REQUIRE(out.find("%r_bpftime_warp0", other) == std::string::npos);
	// Other kernels keep their original stub calls.
	REQUIRE(out.find("call __bpftime_cuda__kernel_trace;", other) !=
		std::string::npos);
	// The compiled hook body is included.
	REQUIRE(out.find(".visible .func __probe_func__test_kernel()") !=
		std::string::npos);
}

TEST_CASE("kprobe_entry warp lowering keeps disabled path identical",
	  "[ptxpass][kprobe_entry]")
{
	const std::string ptx = R"(.version 7.0
.target sm_61
.address_size 64

.visible .entry test_kernel()
{
    mov.u32 %r1, %tid.x;
    call __bpftime_cuda__kernel_trace;
    ret;
}
)";
	auto response =
		run_kprobe_entry_pass(ptx, nlohmann::json::object(), false);
	REQUIRE(response.value("modified", false));
	const std::string out =
		response.value("output_ptx", std::string());
	REQUIRE(out.find("call __probe_func__test_kernel;") !=
		std::string::npos);
	REQUIRE(out.find("%r_bpftime_warp0") == std::string::npos);
	REQUIRE(out.find("activemask") == std::string::npos);
	REQUIRE(out.find("vote.ballot.sync") == std::string::npos);
}

TEST_CASE("End-to-end JSON workflow with empty eBPF",
	  "[ptxpass_core][integration]")
{
	std::string json_input = R"({
  "input":{"full_ptx": ".version 7.0\n.target sm_60\n.visible .entry test() {\n  ret;\n}",
  "to_patch_kernel": "test"},
  "ebpf_instructions": []
})";

	SECTION("parse_runtime_request handles minimal JSON")
	{
		auto request = pass_runtime_request_from_string(json_input);
		REQUIRE(request.input.full_ptx.find(".version 7.0") !=
			std::string::npos);
		REQUIRE(request.input.to_patch_kernel == "test");
		REQUIRE(request.ebpf_instructions.empty());
	}

	SECTION("validation passes for minimal valid PTX")
	{
		nlohmann::json validation = { { "require_entry", true },
					      { "require_ret", true } };

		std::string ptx =
			".version 7.0\n.target sm_60\n.visible .entry test() {\n    ret;\n}";
		REQUIRE(validate_input(ptx, validation));
	}
}

TEST_CASE("compile_ebpf_to_ptx_from_words handles exit instruction",
	  "[ptxpass_core][integration]")
{
	std::vector<uint64_t> exit_only = { 0x0000000000000095ULL };

	std::string result = compile_ebpf_to_ptx_from_words(
		exit_only, "sm_60", "__probe__", true, false);

	REQUIRE_FALSE(result.empty());
	REQUIRE(result.find("ret") != std::string::npos);

	size_t version_count = 0;
	size_t pos = 0;
	while ((pos = result.find(".version", pos)) != std::string::npos) {
		version_count++;
		pos += 8;
	}
	REQUIRE(version_count <= 1);

	size_t target_count = 0;
	pos = 0;
	while ((pos = result.find(".target", pos)) != std::string::npos) {
		target_count++;
		pos += 7;
	}
	REQUIRE(target_count <= 1);
}
