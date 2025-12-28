#include <cstdint>
#include <cctype>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <exception>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <string>
#include <utility>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

struct nv_attach_impl_ptx_compiler {
	std::string error_log;
	std::string info_log;
	std::vector<uint8_t> compiled_program;
};

namespace {

struct temp_file {
	std::string path;
	int fd = -1;
	void reset()
	{
		if (fd >= 0)
			close(fd);
		fd = -1;
		if (!path.empty())
			unlink(path.c_str());
		path.clear();
	}
	~temp_file()
	{
		reset();
	}
	temp_file(const temp_file &) = delete;
	temp_file &operator=(const temp_file &) = delete;
	temp_file() = default;
	temp_file(temp_file &&o) noexcept : path(std::move(o.path)), fd(o.fd)
	{
		o.fd = -1;
	}
	temp_file &operator=(temp_file &&o) noexcept
	{
		if (this == &o)
			return *this;
		reset();
		path = std::move(o.path);
		fd = o.fd;
		o.fd = -1;
		o.path.clear();
		return *this;
	}
};

temp_file make_temp_file(const std::string &prefix)
{
	temp_file f;
	try {
		const auto dir = std::filesystem::temp_directory_path();
		auto tmpl = (dir / (prefix + ".XXXXXX")).string();
		std::vector<char> buf(tmpl.begin(), tmpl.end());
		buf.push_back('\0');
		f.fd = mkstemp(buf.data());
		if (f.fd < 0)
			return {};
		f.path = buf.data();
	} catch (...) {
		return {};
	}
	return f;
}

void rewrite_target_arch_in_place(std::string &ptx, const std::string &gpu_name)
{
	size_t line_start = 0;
	while (line_start < ptx.size()) {
		size_t line_end = ptx.find('\n', line_start);
		const size_t limit =
			(line_end == std::string::npos) ? ptx.size() : line_end;

		size_t i = line_start;
		while (i < limit && (ptx[i] == ' ' || ptx[i] == '\t'))
			i++;

		if (i + 7 <= limit && ptx.compare(i, 7, ".target") == 0 &&
		    (i + 7 == limit || std::isspace((unsigned char)ptx[i + 7]) ||
		     ptx[i + 7] == ',')) {
			size_t pos = i + 7;
			while (pos < limit &&
			       (ptx[pos] == ' ' || ptx[pos] == '\t'))
				pos++;
			if (pos >= limit)
				return;

			size_t end = pos;
			while (end < limit &&
			       !std::isspace((unsigned char)ptx[end]) &&
			       ptx[end] != ',')
				end++;

			ptx.replace(pos, end - pos, gpu_name);
			return;
		}

		if (line_end == std::string::npos)
			break;
		line_start = line_end + 1;
	}
}

int run_ptxas(const std::string &gpu_name, const std::string &in_path,
	      const std::string &out_path, std::string &output)
{
	output.clear();

	int pipefd[2];
	if (pipe(pipefd) != 0)
		return -1;

	const pid_t pid = fork();
	if (pid < 0) {
		close(pipefd[0]);
		close(pipefd[1]);
		return -1;
	}

	if (pid == 0) {
		dup2(pipefd[1], STDOUT_FILENO);
		dup2(pipefd[1], STDERR_FILENO);
		close(pipefd[0]);
		close(pipefd[1]);

		char *argv[] = { (char *)"ptxas",
				 (char *)"--gpu-name",
				 (char *)gpu_name.c_str(),
				 (char *)"-O3",
				 (char *)in_path.c_str(),
				 (char *)"-o",
				 (char *)out_path.c_str(),
				 nullptr };
		execvp(argv[0], argv);
		_exit(127);
	}

	close(pipefd[1]);
	char buf[4096];
	for (;;) {
		const ssize_t n = read(pipefd[0], buf, sizeof(buf));
		if (n == 0)
			break;
		if (n < 0) {
			if (errno == EINTR)
				continue;
			break;
		}
		output.append(buf, buf + n);
	}
	close(pipefd[0]);

	int status = 0;
	if (waitpid(pid, &status, 0) < 0)
		return -1;
	if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
		return 0;
	return -1;
}

} // namespace

extern "C" {
nv_attach_impl_ptx_compiler *nv_attach_impl_create_compiler()
{
	return new nv_attach_impl_ptx_compiler;
}
void nv_attach_impl_destroy_compiler(nv_attach_impl_ptx_compiler *ptr)
{
	delete ptr;
}
int nv_attach_impl_compile(nv_attach_impl_ptx_compiler *ptr, const char *ptx,
			   const char **args, int arg_count)
{
	if (!ptr || !ptx)
		return -1;
	try {
		ptr->error_log.clear();
		ptr->info_log.clear();
		ptr->compiled_program.clear();

		std::string gpu_name;
		for (int i = 0; i < arg_count; i++) {
			if (!args || !args[i])
				continue;
			const char *prefix = "--gpu-name=";
			const size_t prefix_len = strlen(prefix);
			if (strncmp(args[i], prefix, prefix_len) == 0)
				gpu_name = args[i] + prefix_len;
		}
		if (gpu_name.empty()) {
			ptr->error_log = "Missing required option: --gpu-name=<sm_xx> (ptxas-only compiler)";
			return -1;
		}

		auto in_file = make_temp_file("bpftime-ptxas-in");
		if (in_file.fd < 0) {
			ptr->error_log = "mkstemp failed for PTX input";
			return -1;
		}
		auto out_file = make_temp_file("bpftime-ptxas-out");
		if (out_file.fd < 0) {
			ptr->error_log = "mkstemp failed for PTX output";
			return -1;
		}
		close(in_file.fd);
		in_file.fd = -1;
		close(out_file.fd);
		out_file.fd = -1;

		{
			std::ofstream ofs(in_file.path, std::ios::binary);
			if (!ofs) {
				ptr->error_log = "failed to open PTX temp file";
				return -1;
			}
			std::string ptx_s(ptx);
			rewrite_target_arch_in_place(ptx_s, gpu_name);
			ofs << ptx_s;
			ofs.flush();
			if (!ofs) {
				ptr->error_log = "failed to write PTX temp file";
				return -1;
			}
		}

		std::string out;
		if (run_ptxas(gpu_name, in_file.path, out_file.path, out) != 0) {
			ptr->error_log = out.empty() ? "ptxas failed" : out;
			return -1;
		}
		ptr->info_log = out;

		std::ifstream ifs(out_file.path, std::ios::binary);
		ptr->compiled_program.assign(std::istreambuf_iterator<char>(ifs),
					     std::istreambuf_iterator<char>());
		if (ptr->compiled_program.empty()) {
			ptr->error_log = "ptxas produced empty output";
			return -1;
		}
		return 0;
	} catch (const std::exception &e) {
		ptr->error_log = e.what();
		return -1;
	} catch (...) {
		ptr->error_log = "unknown exception";
		return -1;
	}
}

const char *nv_attach_impl_get_error_log(nv_attach_impl_ptx_compiler *ptr)
{
	return ptr ? ptr->error_log.c_str() : "";
}
const char *nv_attach_impl_get_info_log(nv_attach_impl_ptx_compiler *ptr)
{
	return ptr ? ptr->info_log.c_str() : "";
}

int nv_attach_impl_get_compiled_program(nv_attach_impl_ptx_compiler *ptr,
					uint8_t **dest, size_t *size)
{
	if (!ptr || !dest || !size)
		return -1;
	*dest = ptr->compiled_program.data();
	*size = ptr->compiled_program.size();
	return 0;
}
}
