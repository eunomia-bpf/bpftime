#include <ebpf-core.h>
#include <memory>
#include <iostream>
#include <ostream>
#include <fstream>
#include <vector>
#include <cinttypes>
#include <chrono>
#include <optional>

int read_file(const char *path, std::vector<char> &buf)
{
	std::ifstream prog_file(path, std::ios::binary);
	if (!prog_file) {
		std::cerr << "Failed to open " << path << std::endl;
		return 1;
	}
	prog_file.seekg(0, std::ios::end);
	auto file_size = prog_file.tellg();
	prog_file.seekg(0, std::ios::beg);

	buf.resize(file_size);
	if (!prog_file.read(buf.data(), file_size)) {
		std::cerr << "Failed to read " << path << std::endl;
		return 1;
	}
	return 0;
}

int main(int argc, const char **argv)
{
	if (argc < 2 || argc > 3) {
		std::cerr
			<< "Usage: " << argv[0]
			<< " <path to ebpf instructions> [path to memory for the ebpf program]"
			<< std::endl;
		return 1;
	}

	std::vector<char> prog_file;
	if (int err = read_file(argv[1], prog_file); err != 0) {
		return err;
	}
	if (prog_file.size() % 8) {
		std::cerr << "Invalid program size" << std::endl;
		return 1;
	}
	std::optional<std::vector<char> > memory_file;
	if (argc == 3) {
		std::vector<char> buf;
		if (int err = read_file(argv[2], buf); err != 0) {
			return err;
		}
		memory_file = buf;
	}
	std::unique_ptr<ebpf_vm, decltype(&ebpf_destroy)> vm(ebpf_create(),
							     ebpf_destroy);
	char *errmsg = nullptr;
	if (int err = ebpf_load(vm.get(), prog_file.data(),
				(uint32_t)prog_file.size(), &errmsg);
	    err < 0) {
		std::cerr << "Failed to load ebpf program: " << errmsg
			  << std::endl;
		return 1;
	}
	errmsg = nullptr;
	auto compile_start = std::chrono::high_resolution_clock::now();
	auto bpf_main = ebpf_compile(vm.get(), &errmsg);
	auto compile_end = std::chrono::high_resolution_clock::now();

	if (!bpf_main) {
		std::cerr << "Failed to compile ebpf program: " << errmsg
			  << std::endl;
		return 1;
	}
	auto compile_usage =
		std::chrono::duration_cast<std::chrono::nanoseconds>(
			compile_end - compile_start)
			.count();

	auto exec_start = std::chrono::high_resolution_clock::now();
	uint64_t ret = bpf_main(
		memory_file.has_value() ? memory_file->data() : nullptr,
		memory_file.has_value() ? memory_file->size() : 0);

	auto exec_end = std::chrono::high_resolution_clock::now();
	auto exec_usage = std::chrono::duration_cast<std::chrono::nanoseconds>(
				  exec_end - exec_start)
				  .count();

	std::cout << compile_usage << " " << exec_usage << " " << ret
		  << std::endl;
	return 0;
}
