#include "catch2/catch_test_macros.hpp"
#include "ptxpass/core.hpp"
#include <fstream>
#include <iostream>
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
		std::cin.rdbuf(empty_input.rdbuf());
		std::string result = read_all_from_stdin();
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

TEST_CASE("log_transform_stats outputs stats to stderr", "[ptxpass_core]")
{
	std::ostringstream oss;
	std::streambuf *old_cerr = std::cerr.rdbuf(oss.rdbuf());

	log_transform_stats("test_pass", 5, 1024, 2048);

	std::cerr.rdbuf(old_cerr);

	std::string output = oss.str();
	REQUIRE(output.find("test_pass") != std::string::npos);
	REQUIRE(output.find("matched=5") != std::string::npos);
	REQUIRE(output.find("in=1024") != std::string::npos);
	REQUIRE(output.find("out=2048") != std::string::npos);
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
