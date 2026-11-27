// SPDX-License-Identifier: MIT
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/version.h>
#include <hsa/hsa_api_trace.h>
#include <dlfcn.h>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

#define CHECK(condition)                                                       \
	do {                                                                   \
		if (!(condition)) {                                            \
			std::fprintf(stderr, "check failed at line %d: %s\n",  \
				     __LINE__, #condition);                    \
			return 1;                                              \
		}                                                              \
	} while (false)

namespace
{
hsa_queue_t original{}, replacement{};
enum class failure {
	none,
	create,
	register_original,
	allocate,
	register_replacement,
	destroy,
	priority,
	mask
};
failure fail = failure::none;
unsigned original_destroyed, replacement_destroyed, intercepted, written;
const void *written_packets;
uint64_t written_count;
hsa_amd_queue_intercept_handler packet_handler;
void *packet_data;

void host_signal(int)
{
}

hsa_status_t create(hsa_agent_t, uint32_t, hsa_queue_type32_t,
		    void (*)(hsa_status_t, hsa_queue_t *, void *), void *,
		    uint32_t, uint32_t, hsa_queue_t **out)
{
	if (fail == failure::create)
		return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
	*out = &original;
	return HSA_STATUS_SUCCESS;
}
hsa_status_t destroy(hsa_queue_t *queue)
{
	if (queue == &original) {
		if (fail == failure::destroy)
			return HSA_STATUS_ERROR;
		++original_destroyed;
	} else {
		++replacement_destroyed;
	}
	return HSA_STATUS_SUCCESS;
}
hsa_status_t intercept_create(hsa_agent_t, uint32_t, hsa_queue_type32_t,
			      void (*)(hsa_status_t, hsa_queue_t *, void *),
			      void *, uint32_t, uint32_t, hsa_queue_t **out)
{
	++intercepted;
	if (fail == failure::allocate)
		return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
	*out = &replacement;
	return HSA_STATUS_SUCCESS;
}
hsa_status_t intercept_register(hsa_queue_t *queue,
				hsa_amd_queue_intercept_handler cb, void *data)
{
	if (queue == &original)
		return fail == failure::register_original ?
			       HSA_STATUS_ERROR :
			       HSA_STATUS_ERROR_INVALID_QUEUE;
	if (fail == failure::register_replacement)
		return HSA_STATUS_ERROR;
	packet_handler = cb;
	packet_data = data;
	return HSA_STATUS_SUCCESS;
}
void writer(const void *packets, uint64_t count)
{
	++written;
	written_packets = packets;
	written_count = count;
}
#ifdef HSA_AMD_QUEUE_CREATE_DESC_VERSION
hsa_status_t create_desc(hsa_agent_t, hsa_amd_queue_create_desc_t *descs,
			 uint32_t count)
{
	for (uint32_t i = 0; i < count; ++i)
		descs[i].queue = &original;
	return HSA_STATUS_SUCCESS;
}
hsa_status_t priority(hsa_queue_t *, hsa_amd_queue_priority_t)
{
	return fail == failure::priority ? HSA_STATUS_ERROR :
					   HSA_STATUS_SUCCESS;
}
hsa_status_t mask(const hsa_queue_t *, uint32_t, const uint32_t *)
{
	return fail == failure::mask ? HSA_STATUS_ERROR : HSA_STATUS_SUCCESS;
}
#endif
} // namespace

int main(int argc, char **argv)
{
	CHECK(argc == 3);
	// Load the mock by absolute path before the agent: its package RPATH
	// deliberately takes precedence over LD_LIBRARY_PATH during dlmopen.
	void *sdk = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
	CHECK(sdk);
	std::string name = "bpftime-rocm-missing-" + std::to_string(getpid());
	CHECK(setenv("BPFTIME_GLOBAL_SHM_NAME", name.c_str(), 1) == 0);
	struct sigaction action{};
	action.sa_handler = host_signal;
	sigemptyset(&action.sa_mask);
	CHECK(sigaction(SIGUSR1, &action, nullptr) == 0);
	void *agent = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
	CHECK(agent);
	auto configure = reinterpret_cast<decltype(&rocprofiler_configure)>(
		dlsym(agent, "rocprofiler_configure"));
	CHECK(configure);
	rocprofiler_client_id_t client{};
	CHECK(configure(ROCPROFILER_SDK_VERSION, "mock", 0, &client) ==
	      nullptr);
	CHECK(setenv("BPFTIME_ROCM_ATTACH", "1", 1) == 0);
	CHECK(configure(0, "mock", 0, &client) == nullptr);
	CHECK(configure(ROCPROFILER_SDK_VERSION, "mock", 0, nullptr) ==
	      nullptr);
	// Registration failure must leave the host alive and allow a later
	// retry.
	CHECK(configure(ROCPROFILER_SDK_VERSION, "mock", 0, &client) ==
	      nullptr);
	auto allow =
		reinterpret_cast<void (*)()>(dlsym(sdk, "allow_registration"));
	auto registrations = reinterpret_cast<unsigned (*)()>(
		dlsym(sdk, "registration_count"));
	auto tables = reinterpret_cast<void (*)(void **, uint64_t)>(
		dlsym(sdk, "register_test_tables"));
	CHECK(allow && registrations && tables);
	allow();
	auto *config = configure(ROCPROFILER_SDK_VERSION, "mock", 0, &client);
	CHECK(config && config->initialize && config->finalize);
	CHECK(configure(ROCPROFILER_SDK_VERSION, "mock", 0, &client) == config);
	CHECK(registrations() == 1);
	CHECK(config->initialize(nullptr, config->tool_data) == 0);
	struct sigaction after{};
	CHECK(sigaction(SIGUSR1, nullptr, &after) == 0 &&
	      after.sa_handler == host_signal);
	CHECK(getenv("BPFTIME_USED") == nullptr);
	tables(nullptr, 0);
	CoreApiTable core{};
	AmdExtTable amd{};
	HsaApiTable root{};
	root.version.major_id = HSA_API_TABLE_MAJOR_VERSION;
	root.version.minor_id = sizeof(root);
	root.core_ = &core;
	root.amd_ext_ = &amd;
	void *root_ptr = &root;
	tables(&root_ptr, 1); // incomplete table: no callbacks may be installed
	CHECK(core.hsa_queue_create_fn == nullptr);
	core.version.minor_id = sizeof(core);
	core.version.major_id = HSA_CORE_API_TABLE_MAJOR_VERSION;
	amd.version.minor_id = sizeof(amd);
	amd.version.major_id = HSA_AMD_EXT_API_TABLE_MAJOR_VERSION;
	core.hsa_queue_create_fn = create;
	core.hsa_queue_destroy_fn = destroy;
	amd.hsa_amd_queue_intercept_create_fn = intercept_create;
	amd.hsa_amd_queue_intercept_register_fn = intercept_register;
#ifdef HSA_AMD_QUEUE_CREATE_DESC_VERSION
	amd.hsa_amd_queue_create_fn = create_desc;
	amd.hsa_amd_queue_set_priority_fn = priority;
	amd.hsa_amd_queue_cu_set_mask_fn = mask;
#endif
	tables(&root_ptr, 1);
	CHECK(core.hsa_queue_create_fn != create);
	for (auto error : { failure::create, failure::register_original,
			    failure::allocate, failure::register_replacement,
			    failure::destroy, failure::none }) {
		fail = error;
		original_destroyed = replacement_destroyed = intercepted = 0;
		hsa_queue_t *queue = nullptr;
		auto result =
			core.hsa_queue_create_fn({ 0 }, 64,
						 HSA_QUEUE_TYPE_MULTI, nullptr,
						 nullptr, 0, 0, &queue);
		CHECK(result == (error == failure::create ?
					 HSA_STATUS_ERROR_OUT_OF_RESOURCES :
					 HSA_STATUS_SUCCESS));
		if (error == failure::none) {
			CHECK(queue == &replacement &&
			      original_destroyed == 1 &&
			      replacement_destroyed == 0);
		} else if (error != failure::create) {
			CHECK(queue == &original && original_destroyed == 0);
			CHECK(replacement_destroyed ==
			      (error == failure::register_replacement ||
			       error == failure::destroy));
		}
	}
#ifdef HSA_AMD_QUEUE_CREATE_DESC_VERSION
	CHECK(amd.hsa_amd_queue_create_fn != create_desc);
	for (auto error : { failure::priority, failure::mask, failure::none }) {
		fail = error;
		original_destroyed = replacement_destroyed = 0;
		hsa_amd_queue_create_desc_t desc{};
		desc.version = HSA_AMD_QUEUE_CREATE_DESC_VERSION;
		desc.queue_size_bytes = 4096;
		desc.engine_type = HSA_AMD_QUEUE_ENGINE_COMPUTE;
		desc.engine.compute.type = HSA_QUEUE_TYPE_MULTI;
		uint32_t cu_mask = 1;
		desc.engine.compute.cu_mask_count = 32;
		desc.engine.compute.cu_mask = &cu_mask;
		CHECK(amd.hsa_amd_queue_create_fn({ 0 }, &desc, 1) ==
		      HSA_STATUS_SUCCESS);
		CHECK(desc.queue ==
		      (error == failure::none ? &replacement : &original));
		CHECK(original_destroyed == (error == failure::none));
		CHECK(replacement_destroyed == (error != failure::none));
	}
#endif
	hsa_kernel_dispatch_packet_t packets[2]{};
	packets[0].header = HSA_PACKET_TYPE_KERNEL_DISPATCH
			    << HSA_PACKET_HEADER_TYPE;
	packets[0].kernel_object = 123;
	packets[1].header = HSA_PACKET_TYPE_BARRIER_AND
			    << HSA_PACKET_HEADER_TYPE;
	const auto before = packets[0];
	CHECK(packet_handler);
	packet_handler(packets, 2, 0, packet_data, writer);
	CHECK(written == 1 && written_packets == packets && written_count == 2);
	CHECK(std::memcmp(&before, &packets[0], sizeof(before)) == 0);
	config->finalize(config->tool_data);
	config->finalize(config->tool_data);
	intercepted = 0;
	hsa_queue_t *queue = nullptr;
	CHECK(core.hsa_queue_create_fn({ 0 }, 64, HSA_QUEUE_TYPE_MULTI, nullptr,
				       nullptr, 0, 0,
				       &queue) == HSA_STATUS_SUCCESS);
	CHECK(queue == &original && intercepted == 0);
	packet_handler(packets, 2, 0, packet_data, writer);
	CHECK(written == 2);
	CHECK(sigaction(SIGUSR1, nullptr, &after) == 0 &&
	      after.sa_handler == host_signal);
	// Keep the DSO loaded: the runtime owns API/queue callbacks into it.
	return 0;
}
