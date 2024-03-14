#include "attach_private_data.hpp"
#include "bpf_attach_ctx.hpp"
#include "spdlog/cfg/env.h"
#include "spdlog/spdlog.h"
#include <base_attach_impl.hpp>
#include <cerrno>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

using namespace bpftime;
using namespace bpftime::attach;

static const int NGINX_REQUEST_FILTER_ATTACH_TYPE = 2001;

static std::function<int(const char *)> global_url_filter;

class nginx_request_filter_private_data : public attach_private_data {
	friend class nginx_request_filter_attach_impl;
	std::string bad_url;
	int initialize_from_string(const std::string_view &sv)
	{
		bad_url = sv;
		return 0;
	}
};

struct request_filter_argument {
	char url_to_check[128];
	char accept_prefix[128];
};

class nginx_request_filter_attach_impl : public base_attach_impl {
    public:
	nginx_request_filter_attach_impl()
	{
		global_url_filter = [this](const char *url) -> int {
			SPDLOG_INFO("Running callback with {}", url);
			uint64_t ret;
			request_filter_argument arg;
			strcpy(arg.accept_prefix, this->url.c_str());
			strcpy(arg.url_to_check, url);
			int err = this->cb.value()((void *)&arg, sizeof(arg),
						   &ret);
			if (err < 0) {
				SPDLOG_ERROR("Failed to run ebpf program: {}",
					     err);
				return 0;
			}
			return ret;
		};
	}
	std::optional<ebpf_run_callback> cb;
	std::string url;
	int usable_id = -1;
	int detach_by_id(int id)
	{
		if (id != usable_id) {
			SPDLOG_ERROR("Bad id {}", id);
			return -1;
		}
		cb = {};
		usable_id = -1;
		return 0;
	}
	int create_attach_with_ebpf_callback(
		ebpf_run_callback &&cb, const attach_private_data &private_data,
		int attach_type)
	{
		if (attach_type != NGINX_REQUEST_FILTER_ATTACH_TYPE) {
			SPDLOG_ERROR("Unsupported attach type {}", attach_type);
			return -ENOTSUP;
		}
		if (usable_id != -1) {
			SPDLOG_ERROR(
				"Only one nginx url filter could be attached");
			return -EINVAL;
		}
		this->cb = std::move(cb);
		url = dynamic_cast<const nginx_request_filter_private_data &>(
			      private_data)
			      .bad_url;
		usable_id = allocate_id();
		return usable_id;
	}
};

union bpf_attach_ctx_holder {
	bpf_attach_ctx ctx;
	bpf_attach_ctx_holder()
	{
	}
	~bpf_attach_ctx_holder()
	{
	}
	void destroy()
	{
		ctx.~bpf_attach_ctx();
	}
	void init()
	{
		new (&ctx) bpf_attach_ctx;
	}
};
static bpf_attach_ctx_holder ctx_holder;

extern "C" int nginx_plugin_example_run_filter(const char *url)
{
	return global_url_filter(url);
}

extern "C" int nginx_plugin_example_initialize()
{
	spdlog::cfg::load_env_levels();
	bpftime_initialize_global_shm(shm_open_type::SHM_OPEN_ONLY);
	ctx_holder.init();
	auto nginx_req_filter_impl =
		std::make_unique<nginx_request_filter_attach_impl>();
	ctx_holder.ctx.register_attach_impl(
		{ NGINX_REQUEST_FILTER_ATTACH_TYPE },
		std::move(nginx_req_filter_impl),
		[](const std::string_view &sv, int &err) {
			std::unique_ptr<attach_private_data> priv_data =
				std::make_unique<
					nginx_request_filter_private_data>();
			if (int e = priv_data->initialize_from_string(sv);
			    e < 0) {
				err = e;
				return std::unique_ptr<attach_private_data>();
			}
			return priv_data;
		});
	int res = ctx_holder.ctx.init_attach_ctx_from_handlers(
		bpftime_get_agent_config());
	if (res != 0) {
		SPDLOG_ERROR("Failed to initialize attach context: {}", res);
		return res;
	}
	SPDLOG_INFO("Handlers initialized");
	return 0;
}
