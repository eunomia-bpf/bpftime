#include <cerrno>
#include <charconv>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

using print_config_fn = void (*)(int length, char *out);
using process_input_fn = int (*)(const char *input, int length, char *output);

namespace
{
constexpr size_t kMaxOutputBytes = 64U << 20;
constexpr int kResultFd = STDERR_FILENO + 1;

void usage(const char *argv0)
{
	std::cerr << "Usage: " << argv0
		  << " (--config|--process) <pass-library> <output-bytes>\n";
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
#if defined(SYS_close_range)
	if (syscall(SYS_close_range, kResultFd + 1, ~0U, 0) == 0)
		return;
#endif
	long max_fd = sysconf(_SC_OPEN_MAX);
	if (max_fd < 0 || max_fd > std::numeric_limits<int>::max())
		max_fd = 1024;
	for (int fd = kResultFd + 1; fd < max_fd; fd++)
		close(fd);
}

bool write_all(int fd, const char *data, size_t length)
{
	while (length > 0) {
		ssize_t written = write(fd, data, length);
		if (written <= 0) {
			if (written < 0 && errno == EINTR)
				continue;
			if (written == 0)
				errno = EIO;
			return false;
		}
		data += written;
		length -= static_cast<size_t>(written);
	}
	return true;
}
} // namespace

int main(int argc, char **argv)
{
	if (argc != 4) {
		usage(argv[0]);
		return 64;
	}

	const std::string mode = argv[1];
	const char *library_path = argv[2];
	if (mode != "--config" && mode != "--process") {
		usage(argv[0]);
		return 64;
	}
	size_t output_bytes = 0;
	if (!parse_size(argv[3], output_bytes) || output_bytes == 0 ||
	    output_bytes > kMaxOutputBytes) {
		std::cerr << "Invalid output size\n";
		return 64;
	}
	close_inherited_fds();

	std::unique_ptr<void, int (*)(void *)> handle(
		dlopen(library_path, RTLD_NOW | RTLD_LOCAL), dlclose);
	if (!handle) {
		std::cerr << "Unable to load PTX pass " << library_path << ": "
			  << dlerror() << "\n";
		return 66;
	}

	std::vector<char> output(output_bytes, '\0');
	const int output_len = static_cast<int>(output.size());
	int rc = 0;

	if (mode == "--config") {
		auto print_config = reinterpret_cast<print_config_fn>(
			dlsym(handle.get(), "print_config"));
		if (!print_config) {
			std::cerr << "Symbol print_config not found in "
				  << library_path << "\n";
			return 66;
		}
		print_config(output_len, output.data());
	} else {
		auto process_input = reinterpret_cast<process_input_fn>(
			dlsym(handle.get(), "process_input"));
		if (!process_input) {
			std::cerr << "Symbol process_input not found in "
				  << library_path << "\n";
			return 66;
		}
		std::string input = read_stdin();
		rc = process_input(input.c_str(), output_len, output.data());
	}
	handle.reset();
	if (rc != 0)
		return rc;
	if (!write_all(kResultFd, output.data(),
		       nul_terminated_length(output))) {
		std::cerr << "Unable to write PTX pass output: "
			  << std::strerror(errno) << "\n";
		return 70;
	}
	close(kResultFd);
	_exit(0);
}
