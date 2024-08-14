/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */

#include "bpftime_logger.hpp"
#include <cstdlib>
#include <iostream>

void bpftime::bpftime_set_logger_from_env(){
	const char* _target = std::getenv("BPFTIME_LOG_OUTPUT");
	if (_target == NULL){
		bpftime_set_logger("");
	} else {
		bpftime_set_logger(std::string(_target));
	}
}

void bpftime::bpftime_set_logger(const std::string& target){
	if (target == "syslog") {
		throw "Not implemented yet";
	} else if (target == "console" || target == "") { // Default to be console.
		spdlog::debug("Set logger to stderr");
		auto logger = spdlog::stderr_color_mt("stderr");
		logger->set_pattern("[%Y-%m-%d %H:%M:%S][%^%l%$][%t] %v");
		logger->flush_on(spdlog::level::info);
		spdlog::set_default_logger(logger);
	} else {
		spdlog::debug("Set logger to file: {}", target);
		auto logger = spdlog::basic_logger_mt("glogger", target); // Todo. What happens if target invalid || target be wiped?
		logger->set_pattern("[%Y-%m-%d %H:%M:%S][%^%l%$][%t] %v");
		logger->flush_on(spdlog::level::info);
		spdlog::set_default_logger(logger);
	}
	spdlog::cfg::load_env_levels();
}

void bpftime::bpftime_logger_flush(){
	spdlog::default_logger()->flush();
}