#include <catch2/catch_test_macros.hpp>

#include <cerrno>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

extern "C" uint64_t bpftime_trace_printk(uint64_t fmt, uint64_t fmt_size, ...);

namespace {

constexpr const char *TRACEPIPE_PATH_ENV = "TRACEPIPE_PATH";

class scoped_env_var {
      public:
	scoped_env_var(const char *key, const char *value) : key_(key)
	{
		if (const char *old_value = std::getenv(key); old_value != nullptr) {
			had_old_value_ = true;
			old_value_ = old_value;
		}

		if (value != nullptr) {
			if (setenv(key, value, 1) != 0) {
				throw std::runtime_error(
					std::string("setenv failed: ") +
					std::strerror(errno));
			}
		} else if (unsetenv(key) != 0) {
			throw std::runtime_error(std::string("unsetenv failed: ") +
						 std::strerror(errno));
		}
	}

	~scoped_env_var()
	{
		if (had_old_value_) {
			(void)setenv(key_.c_str(), old_value_.c_str(), 1);
		} else {
			(void)unsetenv(key_.c_str());
		}
	}

      private:
	std::string key_;
	bool had_old_value_ = false;
	std::string old_value_;
};

class scoped_temp_dir {
      public:
	scoped_temp_dir()
	{
		auto temp_template =
			(std::filesystem::temp_directory_path() /
			 "bpftime-tracepipe-XXXXXX")
				.string();
		std::vector<char> buffer(temp_template.begin(),
					 temp_template.end());
		buffer.push_back('\0');
		char *created = mkdtemp(buffer.data());
		if (created == nullptr) {
			throw std::runtime_error(
				std::string("mkdtemp failed: ") +
				std::strerror(errno));
		}
		path_ = created;
	}

	~scoped_temp_dir()
	{
		std::error_code ec;
		std::filesystem::remove_all(path_, ec);
	}

	const std::filesystem::path &path() const
	{
		return path_;
	}

      private:
	std::filesystem::path path_;
};

class stdout_capture {
      public:
	stdout_capture()
	{
		int pipe_fds[2];
		if (pipe(pipe_fds) != 0) {
			throw std::runtime_error(
				std::string("pipe failed: ") +
				std::strerror(errno));
		}
		read_fd_ = pipe_fds[0];
		fflush(stdout);
		saved_stdout_ = dup(STDOUT_FILENO);
		if (saved_stdout_ == -1) {
			close(pipe_fds[0]);
			close(pipe_fds[1]);
			throw std::runtime_error(
				std::string("dup failed: ") +
				std::strerror(errno));
		}
		if (dup2(pipe_fds[1], STDOUT_FILENO) == -1) {
			close(pipe_fds[0]);
			close(pipe_fds[1]);
			close(saved_stdout_);
			throw std::runtime_error(
				std::string("dup2 failed: ") +
				std::strerror(errno));
		}
		close(pipe_fds[1]);
	}

	~stdout_capture()
	{
		restore_stdout();
		if (read_fd_ != -1) {
			close(read_fd_);
		}
	}

	std::string finish()
	{
		restore_stdout();
		std::string output;
		char buffer[256];
		while (true) {
			ssize_t ret = read(read_fd_, buffer, sizeof(buffer));
			if (ret > 0) {
				output.append(buffer, ret);
				continue;
			}
			if (ret == -1 && errno == EINTR) {
				continue;
			}
			break;
		}
		close(read_fd_);
		read_fd_ = -1;
		return output;
	}

      private:
	void restore_stdout()
	{
		if (saved_stdout_ == -1) {
			return;
		}
		fflush(stdout);
		(void)dup2(saved_stdout_, STDOUT_FILENO);
		close(saved_stdout_);
		saved_stdout_ = -1;
	}

	int saved_stdout_ = -1;
	int read_fd_ = -1;
};

template <typename Fn> std::string capture_stdout(Fn &&fn)
{
	stdout_capture capture;
	fn();
	return capture.finish();
}

bool is_would_block_error(int error_code)
{
#if EAGAIN == EWOULDBLOCK
	return error_code == EAGAIN;
#else
	return error_code == EAGAIN || error_code == EWOULDBLOCK;
#endif
}

std::string read_fifo_output(int fd, int timeout_ms)
{
	pollfd pfd = {
		.fd = fd,
		.events = POLLIN | POLLHUP,
		.revents = 0,
	};
	while (poll(&pfd, 1, timeout_ms) == -1) {
		if (errno != EINTR) {
			return {};
		}
	}

	std::string output;
	char buffer[256];
	while (true) {
		ssize_t ret = read(fd, buffer, sizeof(buffer));
		if (ret > 0) {
			output.append(buffer, ret);
			continue;
		}
		if (ret == -1 && errno == EINTR) {
			continue;
		}
		if (ret == -1 && is_would_block_error(errno)) {
			break;
		}
		break;
	}
	return output;
}

} // namespace

TEST_CASE("bpftime_trace_printk falls back to stdout when TRACEPIPE_PATH is unset")
{
	scoped_env_var tracepipe_env(TRACEPIPE_PATH_ENV, nullptr);

	auto output = capture_stdout([]() {
		bpftime_trace_printk(
			reinterpret_cast<uint64_t>("stdout fallback %d\n"), 0,
			7);
	});

	REQUIRE(output == "stdout fallback 7\n");
}

TEST_CASE("bpftime_trace_printk writes to tracepipe when a reader is present")
{
	scoped_temp_dir temp_dir;
	auto tracepipe_path = temp_dir.path() / "tracepipe";
	REQUIRE(mkfifo(tracepipe_path.c_str(), 0666) == 0);

	int reader_fd = open(tracepipe_path.c_str(), O_RDONLY | O_NONBLOCK);
	REQUIRE(reader_fd != -1);

	scoped_env_var tracepipe_env(TRACEPIPE_PATH_ENV,
					 tracepipe_path.c_str());
	auto stdout_output = capture_stdout([]() {
		bpftime_trace_printk(
			reinterpret_cast<uint64_t>("fifo path %s %d\n"), 0,
			"ok", 42);
	});

	auto fifo_output = read_fifo_output(reader_fd, 1000);
	close(reader_fd);

	REQUIRE(stdout_output.empty());
	REQUIRE(fifo_output == "fifo path ok 42\n");
}

TEST_CASE("bpftime_trace_printk does not block without a reader and falls back to stdout")
{
	scoped_temp_dir temp_dir;
	auto tracepipe_path = temp_dir.path() / "tracepipe";
	scoped_env_var tracepipe_env(TRACEPIPE_PATH_ENV,
					 tracepipe_path.c_str());

	auto start = std::chrono::steady_clock::now();
	auto output = capture_stdout([]() {
		bpftime_trace_printk(
			reinterpret_cast<uint64_t>("stdout without reader\n"),
			0);
	});
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::steady_clock::now() - start);

	struct stat st = {};
	REQUIRE(lstat(tracepipe_path.c_str(), &st) == 0);
	REQUIRE(S_ISFIFO(st.st_mode));
	REQUIRE(elapsed.count() < 1000);
	REQUIRE(output == "stdout without reader\n");
}

TEST_CASE("bpftime_trace_printk ignores TRACEPIPE_PATH entries that are not FIFOs")
{
	scoped_temp_dir temp_dir;
	auto tracepipe_path = temp_dir.path() / "tracepipe";

	int regular_file_fd =
		open(tracepipe_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
	REQUIRE(regular_file_fd != -1);
	close(regular_file_fd);

	scoped_env_var tracepipe_env(TRACEPIPE_PATH_ENV,
					 tracepipe_path.c_str());
	auto output = capture_stdout([]() {
		bpftime_trace_printk(
			reinterpret_cast<uint64_t>("regular file fallback\n"),
			0);
	});

	struct stat st = {};
	REQUIRE(lstat(tracepipe_path.c_str(), &st) == 0);
	REQUIRE(S_ISREG(st.st_mode));
	REQUIRE(output == "regular file fallback\n");
}
