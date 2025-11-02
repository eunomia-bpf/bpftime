#include "ptxpass_pipeline.hpp"
#include "spdlog/spdlog.h"
#include <boost/process.hpp>
#include <exception>
#include <fstream>
#include "json.hpp"
#include "ptxpass/core.hpp"
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

namespace bpftime::attach
{

using nlohmann::json;

static inline bool is_nonempty_nonblank(const std::string &s)
{
	for (char c : s) {
		if (!(c == ' ' || c == '\n' || c == '\r' || c == '\t' ||
		      c == '\v' || c == '\f'))
			return true;
	}
	return false;
}

std::optional<std::string> run_pass_executable_json(
	const std::string &exec, const std::string &full_ptx,
	const std::string &to_patch_kernel, const std::string &map_sym,
	const std::string &const_sym, const std::vector<ebpf_inst> &ebpf_insts)
{
	SPDLOG_INFO(
		"Running pass executable: {}, to_patch_kernel {}, map_sym {}, const_sym {}",
		exec, to_patch_kernel, map_sym, const_sym);
	using namespace boost::process;
	ipstream child_stdout;
	ipstream child_stderr;
	opstream child_stdin;
	child c(exec, std_out > child_stdout, std_err > child_stderr,
		std_in < child_stdin);
	std::vector<uint64_t> words;
	if (!ebpf_insts.empty()) {
		const uint64_t *words_ptr =
			reinterpret_cast<const uint64_t *>(ebpf_insts.data());
		words.assign(words_ptr, words_ptr + ebpf_insts.size());
	}
	ptxpass::runtime_request::RuntimeRequest req;
	auto &ri = req.input;
	ri.full_ptx = full_ptx;
	ri.to_patch_kernel = to_patch_kernel;
	ri.global_ebpf_map_info_symbol = map_sym;
	ri.ebpf_communication_data_symbol = const_sym;

	req.set_ebpf_instructions(words);
	nlohmann::json in;
	ptxpass::runtime_request::to_json(in, req);
	child_stdin << in.dump();
	child_stdin.flush();
	child_stdin.pipe().close();
	std::ostringstream out, err;
	std::string line;
	while (child_stdout && std::getline(child_stdout, line))
		out << line << '\n';
	while (child_stderr && std::getline(child_stderr, line))
		err << line << '\n';
	c.wait();
	SPDLOG_DEBUG("stdout: {}", out.str());
	SPDLOG_DEBUG("stderr: {}", err.str());

	if (c.exit_code() != 0) {
		SPDLOG_WARN("Program exit abnormally: {}", c.exit_code());
		return std::nullopt;
	}
	const std::string out_str = out.str();
	if (!is_nonempty_nonblank(out_str))
		return std::string();
	try {
		auto j = nlohmann::json::parse(out_str);
		auto ro = j.get<ptxpass::runtime_response::RuntimeResponse>();
		return ro.output_ptx;
	} catch (...) {
		return std::nullopt; // invalid json
	}
}
std::optional<ptxpass::pass_config::PassConfig>
get_pass_config_from_executable(const std::filesystem::path &path)
{
	using namespace ptxpass::pass_config;
	namespace bp = boost::process;
	bp::ipstream pipe_stream;
	bp::child c(path.string() + " --print-config",
		    bp::std_out > pipe_stream);
	std::string output;
	std::string line;
	while (std::getline(pipe_stream, line)) {
		output += line + "\n";
	}

	c.wait();
	try {
		auto input_json = nlohmann::json::parse(output);
		SPDLOG_DEBUG("Got JSON configuration: {}", input_json.dump(4));
		PassConfig config;
		from_json(input_json, config);
		return config;
	} catch (const std::exception &ex) {
		SPDLOG_ERROR("Unable to get configuration: {}", ex.what());
		return {};
	}
}
} // namespace bpftime::attach
