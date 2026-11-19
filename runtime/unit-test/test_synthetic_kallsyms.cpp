/* SPDX-License-Identifier: MIT
 *
 * Tests for the built-in synthetic /proc/kallsyms served in rootless mode.
 */
#include "catch2/catch_test_macros.hpp"

#if __linux__
#include "synthetic_kallsyms.hpp"
#include <string>

TEST_CASE("synthetic /proc/kallsyms table is parseable", "[syscall][kallsyms]")
{
	const std::string &content = bpftime::synthetic_kallsyms_content();
	REQUIRE_FALSE(content.empty());
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
			// "<hex address> <type> <name>"
			size_t first = line.find(' ');
			REQUIRE(first != std::string::npos);
			REQUIRE(line.find(' ', first + 1) != std::string::npos);
			if (line.find(" T _stext") != std::string::npos)
				has_stext = true;
			if (line[0] != '0')
				has_nonzero_text_symbol = true;
		}
		pos = end + 1;
	}
	REQUIRE(lines >= 3);
	REQUIRE(has_stext);
	// A parser that requires at least one real address reads this as
	// "symbols are visible" instead of failing before loading a program.
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

TEST_CASE("only /proc/kallsyms routes to the synthetic table",
	  "[syscall][kallsyms]")
{
	REQUIRE(bpftime::is_synthetic_kallsyms_path("/proc/kallsyms"));
	REQUIRE_FALSE(bpftime::is_synthetic_kallsyms_path("/proc/self/status"));
	REQUIRE_FALSE(
		bpftime::is_synthetic_kallsyms_path("/proc/kallsyms.bak"));
}

TEST_CASE("hidden addresses are detected from a kallsyms sample",
	  "[syscall][kallsyms]")
{
	// kptr_restrict without CAP_SYSLOG: every address is zero.
	const char *hidden = "0000000000000000 T _stext\n"
			     "0000000000000000 T _text\n"
			     "0000000000000000 t do_one_initcall\n";
	REQUIRE(bpftime::kallsyms_addresses_hidden(hidden));
	// A privileged kernel: real addresses must be served as they are.
	const char *visible = "ffffffff81000000 T _stext\n"
			      "ffffffff81000100 T _text\n";
	REQUIRE_FALSE(bpftime::kallsyms_addresses_hidden(visible));
	// One visible module address among hidden core symbols also counts.
	const char *mixed = "0000000000000000 T _stext\n"
			    "ffffffffc0123456 t nf_conntrack_in\t[livepatch]\n";
	REQUIRE_FALSE(bpftime::kallsyms_addresses_hidden(mixed));
}
#endif