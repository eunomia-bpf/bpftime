#include <string>
#include "spdlog/spdlog.h"
#include "spdlog/cfg/env.h"
#include "spdlog/sinks/null_sink.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <cstdlib>
#include <memory>

namespace bpftime
{

inline std::string expand_user_path(const std::string &input_path)
{
	if (input_path.empty()) {
		return {};
	}

	if (input_path[0] == '~') {
		const char *homeDir = getenv("HOME");
		if (!homeDir) {
			return {};
		}

		if (input_path.size() == 1 || input_path[1] == '/') {
			// Replace "~" with the home directory
			std::string expandedPath =
				homeDir + input_path.substr(1);
			return expandedPath;
		} else {
			return {}; // Unsupported path format
		}
	}

	// Return the original path if no tilde expansion is needed
	return input_path;
}

inline void bpftime_set_logger(const std::string &target) noexcept
{
	auto set_quiet_logger = []() noexcept {
		try {
			auto sink =
				std::make_shared<spdlog::sinks::null_sink_mt>();
			auto logger = std::make_shared<spdlog::logger>(
				"bpftime_quiet", sink);
			logger->set_level(spdlog::level::off);
			logger->set_error_handler([](const std::string &) {});
			spdlog::set_default_logger(std::move(logger));
		} catch (...) {
			if (auto *logger = spdlog::default_logger_raw();
			    logger != nullptr)
				logger->set_level(spdlog::level::off);
		}
	};
	try {
		std::string logger_target = expand_user_path(target);
		if (logger_target.empty()) {
			set_quiet_logger();
			return;
		}

		std::shared_ptr<spdlog::sinks::sink> sink;
		if (logger_target == "console") {
			// Console logging is explicitly opt-in and writes to
			// stderr.
			sink = std::make_shared<
				spdlog::sinks::stderr_color_sink_mt>();
		} else {
			constexpr size_t max_size = 1048576 * 5;
			constexpr size_t max_files = 3;
			sink = std::make_shared<
				spdlog::sinks::rotating_file_sink_mt>(
				logger_target, max_size, max_files);
		}
		auto logger = std::make_shared<spdlog::logger>("bpftime_logger",
							       std::move(sink));
		std::weak_ptr<spdlog::logger> weak_logger = logger;
		logger->set_error_handler([weak_logger](const std::string &) {
			if (auto failed_logger = weak_logger.lock())
				failed_logger->set_level(spdlog::level::off);
		});
		logger->set_pattern("[%Y-%m-%d %H:%M:%S][%^%l%$][%t] %v");
		logger->flush_on(spdlog::level::info);
		spdlog::set_default_logger(std::move(logger));

		spdlog::cfg::load_env_levels();
	} catch (...) {
		set_quiet_logger();
	}
}

/*
    Flush the logger.
*/
inline void bpftime_logger_flush() noexcept
{
	try {
		if (auto logger = spdlog::default_logger(); logger != nullptr)
			logger->flush();
	} catch (...) {
	}
}

} // namespace bpftime
