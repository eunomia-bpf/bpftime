#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <unistd.h>
#include <vector>
struct nv_attach_impl_ptx_compiler {
	std::string error_log;
	std::string info_log;
	std::vector<uint8_t> compiled_program;
};
extern "C" {
nv_attach_impl_ptx_compiler *nv_attach_impl_create_compiler()
{
	setlocale(LC_ALL, "");
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
	ptr->error_log.clear(); ptr->info_log.clear(); ptr->compiled_program.clear();
	std::string gpu_name;
	for (int i = 0; i < arg_count; i++) {
		if (!args || !args[i])
			continue;
		const char *prefix = "--gpu-name=";
		if (strncmp(args[i], prefix, strlen(prefix)) == 0) {
			gpu_name = args[i] + strlen(prefix);
		}
	}
	if (gpu_name.empty()) {
		ptr->error_log =
			"Missing required option: --gpu-name=<sm_xx> (ptxas-only compiler)";
		return -1;
	}

	char in_template[] = "/tmp/bpftime-ptxas-in.XXXXXX";
	int in_fd = mkstemp(in_template);
	if (in_fd < 0) {
		ptr->error_log = "mkstemp failed for PTX input";
		return -1;
	}
	close(in_fd);
	char out_template[] = "/tmp/bpftime-ptxas-out.XXXXXX";
	int out_fd = mkstemp(out_template);
	if (out_fd < 0) {
		unlink(in_template);
		ptr->error_log = "mkstemp failed for PTX output";
		return -1;
	}
	close(out_fd);
	{
		std::ofstream ofs(in_template, std::ios::binary);
		std::string ptx_s(ptx);
		auto pos = ptx_s.find(".target");
		if (pos != std::string::npos) {
			pos += strlen(".target");
			while (pos < ptx_s.size() &&
			       (ptx_s[pos] == ' ' || ptx_s[pos] == '\t'))
				pos++;
			auto start = pos;
			while (pos < ptx_s.size() && ptx_s[pos] != ' ' &&
			       ptx_s[pos] != '\t' && ptx_s[pos] != '\n' &&
			       ptx_s[pos] != '\r' && ptx_s[pos] != ',')
				pos++;
			if (pos > start)
				ptx_s.replace(start, pos - start, gpu_name);
		}
		ofs << ptx_s;
	}

	const std::string cmd =
		"ptxas --gpu-name " + gpu_name + " -O3 \"" + in_template +
		"\" -o \"" + out_template + "\" 2>&1";
	int cmd_rc = -1;
	auto out = [&]() -> std::string {
		FILE *fp = popen(cmd.c_str(), "r");
		if (!fp)
			return {};
		std::string out_s;
		char buf[4096];
		while (fgets(buf, sizeof(buf), fp) != nullptr)
			out_s += buf;
		cmd_rc = pclose(fp);
		return out_s;
		}();
	if (cmd_rc != 0 || cmd_rc == -1) {
		ptr->error_log = out.empty() ? "ptxas failed" : out;
		unlink(in_template); unlink(out_template);
		return -1;
	}
	ptr->info_log = out;
	std::ifstream ifs(out_template, std::ios::binary);
	ptr->compiled_program.assign(std::istreambuf_iterator<char>(ifs),
				     std::istreambuf_iterator<char>());
	unlink(in_template); unlink(out_template);
	if (ptr->compiled_program.empty()) {
		ptr->error_log = "ptxas produced empty output";
		return -1;
	}
	return 0;
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
