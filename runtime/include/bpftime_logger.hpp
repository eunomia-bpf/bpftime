#include <string>
#include "spdlog/spdlog.h"
#include "spdlog/cfg/env.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <cstdlib>
#include <iostream>
#include <filesystem>
#include <fstream>

namespace bpftime
{

inline std::string expand_user_path(const std::string &input_path)
{
	if (input_path.empty()) {
		return "console";
	}

	if (input_path[0] == '~') {
		const char *homeDir = getenv("HOME");
		if (!homeDir) {
			return "console";
		}

		if (input_path.size() == 1 || input_path[1] == '/') {
			// Replace "~" with the home directory
			std::string expandedPath = homeDir +
				 input_path.substr(1);
			return expandedPath;
		} else {
			return "console"; // Unsupported path format
		}
	}

	// Return the original path if no tilde expansion is needed
	return input_path;
}

inline void bpftime_set_logger(const std::string &target) noexcept
{
	std::string logger_target = expand_user_path(target);

	if (logger_target == "console") {
		// Set logger to stderr
		auto logger = spdlog::stderr_color_mt("stderr");
		logger->set_pattern("[%Y-%m-%d %H:%M:%S][%^%l%$][%t] %v");
		logger->flush_on(spdlog::level::info);
		spdlog::set_default_logger(logger);
	} else {
		// Set logger to file, with rotation 5MB and 3 files
		auto max_size = 1048576 * 5;
		auto max_files = 3;
		auto logger = spdlog::rotating_logger_mt(
			"bpftime_logger", logger_target, max_size, max_files);
		logger->set_pattern("[%Y-%m-%d %H:%M:%S][%^%l%$][%t] %v");
		logger->flush_on(spdlog::level::info);
		spdlog::set_default_logger(logger);
	}

	// Load log level from environment
	spdlog::cfg::load_env_levels();
}

/*
    Flush the logger.
*/
inline void bpftime_logger_flush()
{
	spdlog::default_logger()->flush();
}

} // namespace bpftime
