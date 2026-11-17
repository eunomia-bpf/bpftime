/* SPDX-License-Identifier: MIT
 *
 * Tests for the built-in synthetic /proc/kallsyms served in rootless mode.
 */
#include "catch2/catch_test_macros.hpp"

#if __linux__
#include "syscall_server_utils.hpp"
#include "synthetic_kallsyms.hpp"
#include <filesystem>
#include <string>

TEST_CASE("synthetic /proc/kallsyms is served as a mocked file",
	  "[syscall][kallsyms]")
{
	auto provider = bpftime::create_mocked_file_based_on_full_path(
		std::filesystem::path("/proc/kallsyms"));
	REQUIRE(provider.has_value());
	const std::string &content = provider->buf;
	// Must be a parseable, non-empty table.
	REQUIRE_FALSE(content.empty());
	// Every line must have at least "addr type name" fields.
	bool has_nonzero_text_symbol = false;
	bool has_stext = false;
	size_t lines = 0;
	size_t pos = 0;
	while (pos < content.size()) {
		size_t end = content.find('\n', pos);
		if (end == std::string::npos)
			end = content.size();
		std::string line = content.substr(pos, end - pos);
		if (!line.empty()) {
			lines++;
			REQUIRE(line.find(' ') != std::string::npos);
			if (line.find(" T _stext") != std::string::npos)
				has_stext = true;
			// A nonzero leading hex address clears the parser's
			// noSymbols condition.
			if (line.size() > 1 && line[0] != '0')
				has_nonzero_text_symbol = true;
		}
		pos = end + 1;
	}
	REQUIRE(lines >= 3);
	REQUIRE(has_stext);
	REQUIRE(has_nonzero_text_symbol);
}

TEST_CASE("synthetic table exposes symbols a generic tracer looks up",
	  "[syscall][kallsyms]")
{
	const std::string &content = bpftime::synthetic_kallsyms_content();
	for (const char *sym :
	     {"_stext", "_etext", "copy_from_user_nofault", "nmi_uaccess_okay",
	      "finish_task_switch"}) {
		REQUIRE(content.find(sym) != std::string::npos);
	}
}

TEST_CASE("unrelated proc files are not mocked as kallsyms",
	  "[syscall][kallsyms]")
{
	auto provider = bpftime::create_mocked_file_based_on_full_path(
		std::filesystem::path("/proc/self/status"));
	if (provider.has_value()) {
		REQUIRE(provider->buf.find("_stext") ==
			std::string::npos);
	}
}
#endif
