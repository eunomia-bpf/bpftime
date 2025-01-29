#ifndef _BPF_ATTACH_CTX
#define _BPF_ATTACH_CTX

#include "attach_private_data.hpp"
#include "base_attach_impl.hpp"
#include "bpftime_config.hpp"
#include "bpftime_helper_group.hpp"
#include "cuda.h"
#include "handler/link_handler.hpp"
#include "handler/perf_event_handler.hpp"
#include "handler/prog_handler.hpp"
#include <atomic>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string_view>
#include <utility>
#include <vector>
typedef struct _GumInterceptor GumInterceptor;
typedef struct _GumInvocationListener GumInvocationListener;

namespace bpftime
{

namespace cuda
{

enum class MapOperation { LOOKUP = 1, UPDATE = 2, DELETE = 3 };

union CallRequest {
	struct {
		char key[256];
	} map_lookup;
	struct {
		char key[256];
		char value[256];
		uint64_t flags;
	} map_update;
	struct {
		char key[256];
	} map_delete;
};

union CallResponse {
	struct {
		int result;
	} map_update, map_delete;
	struct {
		const void *value;
	} map_lookup;
};
/**
 * 我们在这块结构体里放两个标志位和一个简单的参数字段
 * - flag1: device -> host 的信号，“我有请求要处理”
 * - flag2: host   -> device 的信号，“我处理完了”
 * - paramA: 设备端写入的参数，让主机端使用
 */
struct SharedMem {
	int flag1;
	int flag2;
	int occupy_flag;
	int request_id;
	int map_id;
	CallRequest req;
	CallResponse resp;
};
using cuda_context_type = std::remove_pointer<CUcontext>::type;
using cuda_module_type = std::remove_pointer<CUmodule>::type;

void cuda_context_destroyer(CUcontext ptr);
void cuda_module_destroyer(CUmodule ptr);
struct CUDAContext {
	// Indicate whether cuda watcher thread should stop
	std::shared_ptr<std::atomic<bool> > cuda_watcher_should_stop =
		std::make_shared<std::atomic<bool> >(false);

	// Shared memory region for CUDA
	std::unique_ptr<cuda::SharedMem> cuda_shared_mem;
	// Mapped device pointer
	uintptr_t cuda_shared_mem_device_pointer;
	// CUDA context
	std::unique_ptr<cuda_context_type, decltype(&cuda_context_destroyer)>
		ctx_container;
	// Loaded module
	std::optional<std::unique_ptr<cuda_module_type,
				      decltype(&cuda_module_destroyer)> >
		module_container;
	CUDAContext(std::unique_ptr<cuda::SharedMem> &&mem, CUcontext raw_ctx)
		: cuda_shared_mem(std::move(mem)),
		  cuda_shared_mem_device_pointer(
			  (uintptr_t)cuda_shared_mem.get()),
		  ctx_container(raw_ctx, cuda_context_destroyer)
	{
	}
	void set_module(CUmodule raw_ptr)
	{
		module_container.emplace(raw_ptr, cuda_module_destroyer);
	}
};

std::optional<cuda::CUDAContext> create_cuda_context();

} // namespace cuda
class base_attach_manager;

class handler_manager;
class bpftime_prog;

using syscall_hooker_func_t = int64_t (*)(int64_t sys_nr, int64_t arg1,
					  int64_t arg2, int64_t arg3,
					  int64_t arg4, int64_t arg5,
					  int64_t arg6);

class bpf_attach_ctx {
    public:
	bpf_attach_ctx();
	~bpf_attach_ctx();

	// create bpf_attach_ctx from handler_manager in shared memory
	int init_attach_ctx_from_handlers(const handler_manager *manager,
					  const agent_config &config);
	// create bpf_attach_ctx from handler_manager in global_shared_memory
	int init_attach_ctx_from_handlers(const agent_config &config);
	// Register an attach implementation. Attach manager will take its
	// ownership. The third argument is a function that initializes a
	// corresponding attach private data with the given string.
	void register_attach_impl(
		std::initializer_list<int> &&attach_types,
		std::unique_ptr<attach::base_attach_impl> &&impl,
		std::function<std::unique_ptr<attach::attach_private_data>(
			const std::string_view &, int &)>
			private_data_creator);
	// Destroy a specific attach link
	int destroy_instantiated_attach_link(int link_id);
	// Destroy all instantiated attach links
	int destroy_all_attach_links();

    private:
	constexpr static int CURRENT_ID_OFFSET = 65536;
	volatile int current_id = CURRENT_ID_OFFSET;

	// Helpers provided by attach impls
	std::map<int, bpftime_helper_info> helpers;

	// handler_id -> instantiated programs
	std::map<int, std::unique_ptr<bpftime_prog> > instantiated_progs;
	// handler_id -> (instantiated attaches id, attach_impl*)
	std::map<int, std::pair<int, attach::base_attach_impl *> >
		instantiated_attach_links;
	// handler_id -> instantiated attach private data & attach type
	std::map<int,
		 std::pair<std::unique_ptr<attach::attach_private_data>, int> >
		instantiated_perf_events;
	// attach_type -> attach impl
	std::map<int, std::pair<attach::base_attach_impl *,
				std::function<std::unique_ptr<
					attach::attach_private_data>(
					const std::string_view &, int &)> > >
		attach_impls;
	// Holds the ownership of all attach impls
	std::vector<std::unique_ptr<attach::base_attach_impl> >
		attach_impl_holders;
	// Record which handlers were already instantiated
	std::set<int> instantiated_handlers;

	int instantiate_handler_at(const handler_manager *manager, int id,
				   std::set<int> &stk,
				   const agent_config &config);
	int instantiate_prog_handler_at(int id, const bpf_prog_handler &handler,
					const agent_config &config);
	int instantiate_bpf_link_handler_at(int id,
					    const bpf_link_handler &handler);
	int instantiate_perf_event_handler_at(
		int id, const bpf_perf_event_handler &perf_handler);
	// Start host thread for handling map requests from CUDA
	void start_cuda_watcher_thread();
	int start_cuda_program(int id);
	cuda::CUDAContext cuda_ctx;
};

} // namespace bpftime

#endif
