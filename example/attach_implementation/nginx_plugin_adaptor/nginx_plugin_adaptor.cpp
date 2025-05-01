#include "bpf_attach_ctx.hpp"
#include "spdlog/cfg/env.h"
#include "spdlog/spdlog.h"
#include <base_attach_impl.hpp>
#include <functional>
#include <optional>
#include <simple_attach_impl.hpp>
using namespace bpftime;
using namespace bpftime::attach;

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

static const int NGINX_REQUEST_FILTER_ATTACH_TYPE = 2001;

static std::function<int(const std::string &)> trigger_event;
struct request_filter_argument {
	const char *url_to_check;
	const char *accept_prefix;
};

static std::optional<bpf_attach_ctx> ctx_holder;

extern "C" int nginx_plugin_example_run_filter(const char *url)
{
	return trigger_event(url);
}

extern "C" int nginx_plugin_example_initialize()
{
	spdlog::cfg::load_env_levels();
	bpftime_initialize_global_shm(shm_open_type::SHM_OPEN_ONLY);
	ctx_holder.emplace();
	// Save the trigger function to a global variable, after that, we can
	// call it when nginx received any requests.
	trigger_event = simple_attach::add_simple_attach_impl_to_attach_ctx(
		NGINX_REQUEST_FILTER_ATTACH_TYPE,
		[&](const std::string &attach_argument,
		    const std::string &trigger_argument,
		    const attach::ebpf_run_callback &run_ebpf_program) -> int {
			uint64_t ret;
			request_filter_argument arg;
			arg.accept_prefix = attach_argument.c_str();
			arg.url_to_check = trigger_argument.c_str();
			SPDLOG_DEBUG("accept prefix={}, url to check = {}",
				     arg.accept_prefix, arg.url_to_check);
			int err = run_ebpf_program((void *)&arg, sizeof(arg),
						   &ret);
			if (unlikely(err < 0)) {
				SPDLOG_ERROR("Failed to run ebpf program: {}",
					     err);
				return 0;
			}
			return ret != 0;
		},
		*ctx_holder);

	int res = ctx_holder->init_attach_ctx_from_handlers(
		bpftime_get_agent_config());
	if (res != 0) {
		SPDLOG_ERROR("Failed to initialize attach context: {}", res);
		return res;
	}
	SPDLOG_INFO("Handlers initialized");
	return 0;
}
