#include "spdlog/spdlog.h"
#include "spdlog/cfg/env.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace bpftime {

void bpftime_set_logger_from_env();

/*
	Set global logger for this process.
	- "console": Default. Output to stderr. Suggested for temporary debugging.
	- "syslog": Output to syslog. Suggested for daemon as deployed service.
	- some file path. Output to file.
*/
void bpftime_set_logger(const std::string& target);

void bpftime_logger_flush();
	
}