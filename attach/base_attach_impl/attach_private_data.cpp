#include "attach_private_data.hpp"
#include "spdlog/spdlog.h"
#include <stdexcept>
using namespace bpftime::attach;

attach_private_data::~attach_private_data()
{
}
int attach_private_data::initialize_from_string(const std::string_view &sv)
{
	SPDLOG_ERROR(
		"Not implemented: attach_private_data::initialize_from_string");
	throw std::runtime_error("attach_private_data::initialize_from_string");
}
