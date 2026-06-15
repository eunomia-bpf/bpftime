#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

using print_config_fn = void (*)(int length, char *out);
using process_input_fn = int (*)(const char *input, int length, char *output);

namespace
{
constexpr size_t kDefaultConfigOutputBytes = 256U << 10;
constexpr size_t kDefaultProcessOutputBytes = 1U << 20;
constexpr size_t kMaxOutputBytes = 64U << 20;

void usage(const char *argv0)
{
	std::cerr << "Usage: " << argv0
		  << " (--config|--process) <pass-library> [--output-bytes N]\n";
}

bool parse_size(const char *text, size_t &out)
{
	const char *end = text + std::strlen(text);
	auto [ptr, ec] = std::from_chars(text, end, out);
	return ec == std::errc() && ptr == end;
}

std::string read_stdin()
{
	return std::string(std::istreambuf_iterator<char>(std::cin),
			   std::istreambuf_iterator<char>());
}

size_t nul_terminated_length(const std::vector<char> &buffer)
{
	const void *nul = std::memchr(buffer.data(), '\0', buffer.size());
	if (nul == nullptr)
		return buffer.size();
	return static_cast<const char *>(nul) - buffer.data();
}

void close_inherited_fds()
{
#if defined(__linux__)
	DIR *dir = opendir("/proc/self/fd");
	if (dir != nullptr) {
		int self_fd = dirfd(dir);
		std::vector<int> fds;
		for (dirent *entry = readdir(dir); entry != nullptr;
		     entry = readdir(dir)) {
			char *end = nullptr;
			errno = 0;
			long fd = std::strtol(entry->d_name, &end, 10);
			if (errno != 0 || end == entry->d_name || *end != '\0' ||
			    fd <= STDERR_FILENO || fd == self_fd ||
			    fd > std::numeric_limits<int>::max())
				continue;
			fds.push_back(static_cast<int>(fd));
		}
		closedir(dir);
		for (int fd : fds)
			close(fd);
		return;
	}
#endif
	long max_fd = sysconf(_SC_OPEN_MAX);
	if (max_fd < 0)
		max_fd = 1024;
	if (max_fd > std::numeric_limits<int>::max())
		max_fd = std::numeric_limits<int>::max();
	for (int fd = STDERR_FILENO + 1; fd < max_fd; fd++)
		close(fd);
}
} // namespace

int main(int argc, char **argv)
{
	if (argc < 3) {
		usage(argv[0]);
		return 64;
	}

	const std::string mode = argv[1];
	const char *library_path = argv[2];
	if (mode != "--config" && mode != "--process") {
		usage(argv[0]);
		return 64;
	}
	size_t output_bytes = kDefaultProcessOutputBytes;
	if (mode == "--config")
		output_bytes = kDefaultConfigOutputBytes;
	for (int i = 3; i < argc; i++) {
		if (std::strcmp(argv[i], "--output-bytes") == 0 &&
		    i + 1 < argc) {
			if (!parse_size(argv[++i], output_bytes) ||
			    output_bytes == 0 || output_bytes > kMaxOutputBytes ||
			    output_bytes >
				    (size_t)std::numeric_limits<int>::max()) {
				std::cerr << "Invalid --output-bytes value\n";
				return 64;
			}
			continue;
		}
		usage(argv[0]);
		return 64;
	}
	close_inherited_fds();

	fflush(stdout);
	int saved_stdout = dup(STDOUT_FILENO);
	if (saved_stdout == -1 || dup2(STDERR_FILENO, STDOUT_FILENO) == -1) {
		std::cerr << "Unable to redirect pass stdout: "
			  << std::strerror(errno) << "\n";
		if (saved_stdout != -1)
			close(saved_stdout);
		return 70;
	}
	auto restore_stdout = [&]() -> bool {
		std::cout.flush();
		std::clog.flush();
		std::cerr.flush();
		fflush(stdout);
		if (dup2(saved_stdout, STDOUT_FILENO) == -1) {
			std::cerr << "Unable to restore stdout: "
				  << std::strerror(errno) << "\n";
			close(saved_stdout);
			saved_stdout = -1;
			return false;
		}
		close(saved_stdout);
		saved_stdout = -1;
		return true;
	};

	std::unique_ptr<void, int (*)(void *)> handle(
		dlopen(library_path, RTLD_NOW | RTLD_LOCAL), dlclose);
	if (!handle) {
		std::cerr << "Unable to load PTX pass " << library_path << ": "
			  << dlerror() << "\n";
		return 66;
	}

	std::vector<char> output(output_bytes, '\0');
	const int output_len = static_cast<int>(output.size());

	if (mode == "--config") {
		auto print_config =
			reinterpret_cast<print_config_fn>(
				dlsym(handle.get(), "print_config"));
		if (!print_config) {
			std::cerr << "Symbol print_config not found in "
				  << library_path << "\n";
			return 66;
		}
		print_config(output_len, output.data());
		if (!restore_stdout())
			return 70;
		std::cout.write(output.data(), nul_terminated_length(output));
		return 0;
	}

	if (mode == "--process") {
		auto process_input =
			reinterpret_cast<process_input_fn>(
				dlsym(handle.get(), "process_input"));
		if (!process_input) {
			std::cerr << "Symbol process_input not found in "
				  << library_path << "\n";
			return 66;
		}
		std::string input = read_stdin();
		int rc = process_input(input.c_str(), output_len, output.data());
		if (!restore_stdout())
			return 70;
		if (rc != 0)
			return rc;
		std::cout.write(output.data(), nul_terminated_length(output));
		return 0;
	}

	usage(argv[0]);
	return 64;
}
