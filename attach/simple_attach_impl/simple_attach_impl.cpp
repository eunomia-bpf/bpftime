#include "simple_attach_impl.hpp"
#include "spdlog/spdlog.h"

using namespace bpftime::simple_attach;
using namespace bpftime;

int simple_attach_impl::create_attach_with_ebpf_callback(
	attach::ebpf_run_callback &&cb,
	const attach::attach_private_data &private_data, int attach_type)
{
	if (usable_id != -1) {
		SPDLOG_ERROR("Simple attach impl only supports one instance");
		return -1;
	}
	if (attach_type != predefined_attach_type) {
		SPDLOG_ERROR(
			"Trying to attach a simple_attach_impl with mismatched attach type, expected = {}, tried = {}",
			predefined_attach_type, attach_type);
		return -1;
	}
	auto &attach_private_data =
		dynamic_cast<const simple_attach_private_data &>(private_data);
	this->argument = attach_private_data.data;
	this->ebpf_callback = cb;
	usable_id = allocate_id();
	SPDLOG_DEBUG("Registered through simple attach impl, type = {}",
		     attach_type);
	return usable_id;
}

int simple_attach_impl::detach_by_id(int id)
{
	if (usable_id == -1) {
		SPDLOG_ERROR("Not attached yet");
		return -1;
	}
	if (usable_id != id) {
		SPDLOG_ERROR("Trying to detach an invalid id");
		return -1;
	}
	usable_id = -1;
	return 0;
}
int simple_attach_impl::trigger()
{
	if (usable_id == -1) {
		SPDLOG_WARN(
			"Triggering unregistered simple_attach_impl type {}",
			predefined_attach_type);
		return 01;
	}
	SPDLOG_DEBUG("Triggering simple_attach_impl with type {}",
		     predefined_attach_type);
	return this->callback(argument, ebpf_callback);
}
