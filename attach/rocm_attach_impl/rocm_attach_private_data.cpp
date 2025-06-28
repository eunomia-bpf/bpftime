#include "rocm_attach_private_data.hpp"
#include "spdlog/spdlog.h"

int bpftime::attach::rocm_attach_private_data::initialize_from_string(const std::string_view &sv)
{
	SPDLOG_INFO(
		"Calling rocm_attach_private_data::initialize_from_string, which is a no-op");
	return 0;
}
