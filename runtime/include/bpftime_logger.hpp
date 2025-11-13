#include <string>
#include <memory>
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
			std::string expandedPath =
				homeDir + input_path.substr(1);
			return expandedPath;
		} else {
		// Create or reuse a named rotating file logger per target path.
			return "console"; // Unsupported path format
		}
	}

	// Return the original path if no tilde expansion is needed
	return input_path;
}

inline void bpftime_set_logger(const std::string &target) noexcept
{
	std::string logger_target = expand_user_path(target);
	std::shared_ptr<spdlog::logger> logger;
	if (logger_target == "console") {
		// Reuse the same console logger across initializations to avoid duplicate-name errors.
		constexpr const char *logger_name = "bpftime_console";
		logger = spdlog::get(logger_name);
		if (!logger) {
			logger = spdlog::stderr_color_mt(logger_name);
		}
	} else {
		const std::string logger_name =
			std::string("bpftime_file_") + logger_target;
		logger = spdlog::get(logger_name);
		if (!logger) {
			auto max_size = 1048576 * 5;
			auto max_files = 3;
			logger = spdlog::rotating_logger_mt(
				logger_name, logger_target, max_size, max_files);
		}
	}
	logger->set_pattern("[%Y-%m-%d %H:%M:%S][%^%l%$][%t] %v");
	logger->flush_on(spdlog::level::info);
	spdlog::set_default_logger(logger);

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
