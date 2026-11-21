#include <cerrno>
#include <charconv>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <iterator>
#include <limits>
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
	if (argc != 4 || (std::strcmp(argv[1], "--config") != 0 &&
			  std::strcmp(argv[1], "--process") != 0)) {
		std::cerr
			<< "Usage: " << argv[0]
			<< " (--config|--process) <pass-library> <output-bytes>\n";
		return 64;
	}
	const char *library_path = argv[2];
	size_t output_bytes = 0;
	const char *size_end = argv[3] + std::strlen(argv[3]);
	auto [size_ptr, size_error] =
		std::from_chars(argv[3], size_end, output_bytes);
	if (size_error != std::errc() || size_ptr != size_end ||
	    output_bytes == 0 || output_bytes > kMaxOutputBytes) {
		std::cerr << "Invalid output size\n";
		return 64;
	}
	close_inherited_fds();

	void *handle = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
	if (!handle) {
		std::cerr << "Unable to load PTX pass " << library_path << ": "
			  << dlerror() << "\n";
		return 66;
	}

	std::vector<char> output(output_bytes, '\0');
	const int output_len = static_cast<int>(output.size());
	int rc = 0;
	auto print_config = reinterpret_cast<print_config_fn>(
		dlsym(handle, "print_config"));
	auto process_input = reinterpret_cast<process_input_fn>(
		dlsym(handle, "process_input"));
	if (!print_config || !process_input) {
		std::cerr << "Required PTX pass symbols not found in "
			  << library_path << "\n";
		return 66;
	}

	if (std::strcmp(argv[1], "--config") == 0) {
		print_config(output_len, output.data());
	} else {
		std::string input{ std::istreambuf_iterator<char>(std::cin),
				   std::istreambuf_iterator<char>() };
		rc = process_input(input.c_str(), output_len, output.data());
	}
	dlclose(handle);
	if (rc != 0)
		return rc;
	if (!write_all(kResultFd, output.data(),
		       strnlen(output.data(), output.size()))) {
		std::cerr << "Unable to write PTX pass output: "
			  << std::strerror(errno) << "\n";
		return 70;
	}
	close(kResultFd);
	_exit(0);
}
