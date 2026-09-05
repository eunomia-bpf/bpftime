#include "sass_aot.hpp"
#include "sass_aot_test_config.hpp"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
	using namespace bpftime::attach::sass_aot;
	if (argc > 3) {
		std::cerr << "usage: " << argv[0]
			  << " [artifact-directory] [device-ordinal]\n";
		return 2;
	}

	SassAotOptions compile_options;
	compile_options.out_dir =
		argc >= 2 ? argv[1] : "/tmp/bpftime_sass_aot_live";
	compile_options.func_name = "sass_aot_probe";
	compile_options.context_size = sizeof(uint64_t);

	std::vector<uint64_t> words;
	std::string section;
	if (const auto error =
		    load_bpf_program_words(SASS_AOT_SPIKE_BPF_OBJECT,
					   "cuda__/sass_aot", words, section)) {
		std::cerr << *error << '\n';
		return 1;
	}

	const auto compiled =
		compile_ebpf_to_sass_aot(words, section, compile_options);
	if (!compiled.ok) {
		std::cerr << compiled.error << '\n';
		return 1;
	}

	SassAotExecutionOptions execution_options;
	if (argc == 3) {
		try {
			execution_options.device_ordinal = std::stoi(argv[2]);
		} catch (...) {
			std::cerr << "device ordinal must be an integer\n";
			return 2;
		}
	}
	std::vector<uint8_t> context(compiled.context_size, 0);
	const auto executed =
		execute_sass_aot(compiled, context, execution_options);
	if (!executed.ok) {
		std::cerr << executed.error << '\n';
		return 1;
	}

	uint64_t value = 0;
	std::memcpy(&value, context.data(), sizeof(value));
	if (value != 42) {
		std::cerr << "unexpected SASS result: " << value
			  << " (expected 42)\n";
		return 1;
	}
	std::cout << "verified SASS result: " << value << '\n';
	std::cout << "cubin: " << compiled.cubin_path << '\n';
	return 0;
}
