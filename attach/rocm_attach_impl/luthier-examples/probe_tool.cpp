// SPDX-License-Identifier: MIT
// Command-line verifier for the separately compiled HIP workload and tool.
#include <json.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;
namespace fs = std::filesystem;
using json = nlohmann::json;

// Spawn the separately compiled HIP fixture with the tool configured before
// HIP loads, then wait for tool finalization before validating its reports.
static void run_child(const fs::path &application, const fs::path &tool,
		      const std::string &mode, const fs::path &output)
{
	std::vector<std::string> environment;
	for (char **entry = environ; *entry; ++entry) {
		std::string value(*entry);
		const auto name = value.substr(0, value.find('='));
		if (name != "LD_PRELOAD" && name != "ROCP_TOOL_LIBRARIES" &&
		    name != "LUTHIER_ARGS" &&
		    name != "BPFTIME_LOG_OUTPUT" && name != "SPDLOG_LEVEL" &&
		    name != "BPFTIME_GLOBAL_SHM_NAME" &&
		    name != "BPFTIME_LUTHIER_REPORT" &&
		    name != "BPFTIME_LUTHIER_MODE")
			environment.push_back(std::move(value));
	}
	if (mode == "agent-preload") {
		environment.push_back("LD_PRELOAD=" + tool.string());
		// Exercise failed initialization in a real HIP host, isolated from
		// any live bpftime session. The workload must still run quietly.
		environment.push_back(
			"BPFTIME_GLOBAL_SHM_NAME=bpftime-luthier-probe-" +
			std::to_string(getpid()) + "-" +
			std::to_string(std::chrono::steady_clock::now()
					       .time_since_epoch().count()));
	} else if (mode != "baseline") {
		environment.push_back("ROCP_TOOL_LIBRARIES=" + tool.string());
	}
	if (mode != "baseline" && mode != "unconfigured" &&
	    mode != "agent-preload") {
		auto report_path = output / "probe.jsonl";
		if (mode == "bad-report") {
			// A regular file as a parent forces fopen to fail with
			// ENOTDIR.
			std::ofstream invalid(output / "not-a-directory");
			invalid.exceptions(std::ios::failbit |
					   std::ios::badbit);
			invalid << "fixture\n";
			invalid.close();
			report_path =
				output / "not-a-directory" / "probe.jsonl";
		}
		environment.push_back("BPFTIME_LUTHIER_REPORT=" +
				      report_path.string());
		environment.push_back(
			"BPFTIME_LUTHIER_MODE=" +
			(mode == "bad-report" ? std::string("entry") : mode));
	}
	std::vector<char *> env;
	for (auto &value : environment)
		env.push_back(value.data());
	env.push_back(nullptr);
	auto executable = application.string();
	char *args[] = { executable.data(), nullptr };
	auto check_spawn = [](int error) {
		if (error)
			throw std::system_error(error, std::generic_category(),
						"spawn workload");
	};
	struct spawn_actions {
		posix_spawn_file_actions_t value;
		spawn_actions()
		{
			if (const int error =
				    posix_spawn_file_actions_init(&value))
				throw std::system_error(error,
							std::generic_category(),
							"spawn actions");
		}
		~spawn_actions()
		{
			posix_spawn_file_actions_destroy(&value);
		}
	};
	// File actions affect only our explicitly launched test child, never
	// the injected tool's host process or this runner's standard streams.
	spawn_actions guard;
	check_spawn(posix_spawn_file_actions_addopen(
		&guard.value, STDOUT_FILENO, (output / "stdout.log").c_str(),
		O_WRONLY | O_CREAT | O_TRUNC, 0600));
	check_spawn(posix_spawn_file_actions_addopen(
		&guard.value, STDERR_FILENO, (output / "stderr.log").c_str(),
		O_WRONLY | O_CREAT | O_TRUNC, 0600));
	pid_t child;
	check_spawn(posix_spawn(&child, executable.c_str(), &guard.value,
				nullptr, args, env.data()));
	const auto deadline =
		std::chrono::steady_clock::now() + std::chrono::seconds(150);
	int status = 0;
	for (;;) {
		const auto result = waitpid(child, &status, WNOHANG);
		if (result == child)
			break;
		if (result < 0 && errno != EINTR)
			throw std::system_error(errno, std::generic_category(),
						"waitpid");
		if (std::chrono::steady_clock::now() >= deadline) {
			kill(child, SIGKILL);
			while (waitpid(child, &status, 0) < 0 &&
			       errno == EINTR) {
			}
			throw std::runtime_error("Workload timed out; see " +
						 output.string());
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
		throw std::runtime_error("Workload failed (wait status " +
					 std::to_string(status) + "); see " +
					 output.string());
}

static json verify(const std::string &mode, const fs::path &output)
{
	auto require = [](bool condition, const std::string &message) {
		if (!condition)
			throw std::runtime_error(message);
	};
	std::ifstream stdout_log(output / "stdout.log");
	std::string line, extra;
	require(static_cast<bool>(std::getline(stdout_log, line)),
		"Missing workload result");
	require(!std::getline(stdout_log, extra) &&
			fs::file_size(output / "stderr.log") == 0,
		"Unexpected host stdio output; see " + output.string());
	const auto app = json::parse(line);
	require(app.at("status") == "pass" && app.at("launches") == 12 &&
			app.at("checksum") == 2349641562ULL,
		"Incorrect workload result");
	const auto wave_size = app.at("wave_size").get<uint64_t>();
	require(wave_size == 32 || wave_size == 64, "Unexpected wave size");
	const auto report_path = output / "probe.jsonl";
	size_t events = 0;
	if (mode == "baseline" || mode == "unconfigured" ||
	    mode == "bad-report" || mode == "agent-preload") {
		require(!fs::exists(report_path),
			"Inactive tool unexpectedly wrote a report");
	} else {
		std::ifstream report(report_path);
		require(report.is_open(),
			"Luthier did not initialize: missing probe report");
		bool initialized = false, finalized = false;
		uint64_t dispatches = 0, total_waves = 0;
		while (std::getline(report, line)) {
			const auto event = json::parse(line);
			const auto status =
				event.at("status").get<std::string>();
			++events;
			if (status == "initialized") {
				require(events == 1 && event.at("mode") == mode,
					"Unexpected initialization event");
				initialized = true;
			} else if (status == "finalized") {
				require(initialized && !finalized,
					"Unexpected finalization event");
				finalized = true;
			} else if (status == "dispatch") {
				require(initialized && !finalized,
					"Dispatch outside tool lifetime");
				const auto block =
					event.at("block_x").get<uint64_t>();
				const auto grid =
					event.at("grid_x").get<uint64_t>();
				require((block == 64 || block == 128) && grid &&
						grid % block == 0,
					"Unexpected launch dimensions");
				const auto waves =
					grid / block *
					((block + wave_size - 1) / wave_size);
				const uint64_t entry =
					mode == "roundtrip" ? 0 : waves * 7;
				const uint64_t exit =
					mode == "entry-exit" ? waves * 7 : 0;
				require(event.at("entry") == entry &&
						event.at("exit") == exit &&
						event.at("weight") == 7,
					"Incorrect device counters: " +
						event.dump());
				total_waves += waves;
				++dispatches;
			} else {
				throw std::runtime_error(
					"Luthier error or unexpected event: " +
					event.dump());
			}
		}
		require(report.eof(), "Cannot read probe report");
		require(initialized && finalized,
			"Missing initialization/finalization event");
		require(dispatches == app.at("launches") &&
				total_waves == app.at("waves"),
			"Observed dispatches do not match workload");
	}
	return { { "mode", mode },
		 { "result", "pass" },
		 { "application", app },
		 { "probe_events", events } };
}

int main(int argc, char **argv)
{
	try {
		std::vector<const char *> modes{ "baseline", "unconfigured",
			"bad-report", "roundtrip", "entry", "entry-exit" };
		std::string mode = "all";
		fs::path application = BPFTIME_LUTHIER_APPLICATION;
		fs::path tool = BPFTIME_LUTHIER_TOOL_LIBRARY;
		fs::path agent = BPFTIME_LUTHIER_AGENT_LIBRARY;
		fs::path output = BPFTIME_LUTHIER_RESULTS;
		for (int i = 1; i < argc; i += 2) {
			const std::string option(argv[i]);
			if (option == "--help") {
				std::cout
					<< "Usage: bpftime_luthier_probe [--mode all|baseline|unconfigured|bad-report|roundtrip|entry|entry-exit|agent-preload]"
					   " [--output-dir DIR] [--application HIP_EXECUTABLE] [--tool TOOL_DSO] [--agent AGENT_DSO]\n";
				return 0;
			}
			if (i + 1 == argc)
				throw std::runtime_error(
					"Missing option value");
			if (option == "--mode")
				mode = argv[i + 1];
			else if (option == "--application")
				application = argv[i + 1];
			else if (option == "--tool")
				tool = argv[i + 1];
			else if (option == "--agent")
				agent = argv[i + 1];
			else if (option == "--output-dir")
				output = argv[i + 1];
			else
				throw std::runtime_error("Unknown option: " +
							 option);
		}
		if (!agent.empty()) {
			if (!fs::is_regular_file(agent))
				throw std::runtime_error("Missing agent: " +
							 agent.string());
			agent = fs::absolute(agent);
			modes.push_back("agent-preload");
		} else if (mode == "agent-preload") {
			throw std::runtime_error("agent-preload requires --agent AGENT_DSO or BPFTIME_ENABLE_ROCM_ATTACH=ON");
		}
		if (mode != "all" &&
		    std::find(modes.begin(), modes.end(), mode) == modes.end())
			throw std::runtime_error("Unknown mode: " + mode);
		if (!fs::is_regular_file(application))
			throw std::runtime_error("Missing HIP executable: " +
						 application.string());
		if (!fs::is_regular_file(tool))
			throw std::runtime_error("Missing probe tool: " +
						 tool.string());
		if (output.empty())
			throw std::runtime_error(
				"Output directory must not be empty");
		application = fs::absolute(application);
		tool = fs::absolute(tool);
		output = fs::absolute(output);
		for (const auto *current : modes) {
			if (mode != "all" && mode != current)
				continue;
			const auto directory = output / current;
			fs::create_directories(directory);
			fs::remove(directory / "probe.jsonl");
			fs::remove(directory / "summary.json");
			run_child(application,
				  std::string(current) == "agent-preload" ? agent : tool,
				  current, directory);
			const auto summary = verify(current, directory);
			std::ofstream result(directory / "summary.json");
			result.exceptions(std::ios::failbit | std::ios::badbit);
			result << summary.dump(2) << '\n';
			result.close();
			std::cout << summary.dump() << '\n';
		}
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "FAIL: " << error.what() << '\n';
		return 1;
	}
}
