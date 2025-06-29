#include "rocm_attach_private_data.hpp"
#include "spdlog/spdlog.h"

int bpftime::attach::rocm_attach_private_data::initialize_from_string(
	const std::string_view &sv)
{
	SPDLOG_INFO("Initializing rocm attach private data from {}", sv);
	if (sv.starts_with("rocm:exit:")) {
		this->is_ret_probe = true;
		this->func_name = sv.substr(10);
	} else if (sv.starts_with("rocm:entry:")) {
		this->is_ret_probe = false;
		this->func_name = sv.substr(11);
	} else {
		SPDLOG_ERROR(
			"Invalid string to initialize rocm attach private data: {}",
			sv);
		return -1;
	}
	SPDLOG_INFO(
		"Initialized rocm attach private data: ret probe={}, func_name={}",
		this->is_ret_probe, this->func_name);
	return 0;
}
