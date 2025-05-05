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
#include <array>
#include <atomic>
#include <cstdint>
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

enum class HelperOperation {
	MAP_LOOKUP = 1,
	MAP_UPDATE = 2,
	MAP_DELETE = 3,
	MAP_GET_NEXT_KEY = 4,
	TRACE_PRINTK = 6
};

union HelperCallRequest {
	struct {
		char key[1 << 30];
	} map_lookup;
	struct {
		char key[1 << 30];
		char value[1 << 30];
		uint64_t flags;
	} map_update;
	struct {
		char key[1 << 30];
	} map_delete;

	struct {
		char fmt[1000];
		int fmt_size;
		unsigned long arg1, arg2, arg3;
	} trace_printk;
};

union HelperCallResponse {
	struct {
		int result;
	} map_update, map_delete, trace_printk;
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
struct CommSharedMem {
	int flag1;
	int flag2;
	int occupy_flag;
	int request_id;
	long map_id;
	HelperCallRequest req;
	HelperCallResponse resp;
	uint64_t time_sum[8];
};

struct MapBasicInfo {
	bool enabled;
	int key_size;
	int value_size;
	int max_entries;
};

using cuda_context_type = std::remove_pointer<CUcontext>::type;
using cuda_module_type = std::remove_pointer<CUmodule>::type;

void cuda_context_destroyer(CUcontext ptr);
void cuda_module_destroyer(CUmodule ptr);

struct CUDAProgramRecord {
	std::string probe_func;
	int prog_id;
};
struct CUDAContext {
	// Indicate whether cuda watcher thread should stop
	std::shared_ptr<std::atomic<bool> > cuda_watcher_should_stop =
		std::make_shared<std::atomic<bool> >(false);

	// Shared memory region for CUDA
	std::unique_ptr<cuda::CommSharedMem> cuda_shared_mem;
	// Mapped device pointer
	uintptr_t cuda_shared_mem_device_pointer;
	// CUDA context
	std::unique_ptr<cuda_context_type, decltype(&cuda_context_destroyer)>
		ctx_container;
	// Loaded module
	std::optional<std::unique_ptr<cuda_module_type,
				      decltype(&cuda_module_destroyer)> >
		probe_module_container;
	std::optional<std::unique_ptr<cuda_module_type,
				      decltype(&cuda_module_destroyer)> >
		demo_module_container;

	std::unique_ptr<std::array<std::atomic<uint64_t>, 8> >
		operation_time_sum;

	std::unique_ptr<std::array<int32_t, 10> > demo_prog_array;
	std::unique_ptr<int64_t> demo_prog_sum_out;

	CUDAContext(std::unique_ptr<cuda::CommSharedMem> &&mem,
		    CUcontext raw_ctx,
		    std::unique_ptr<std::array<int32_t, 10> > &&demo_prog_array,
		    std::unique_ptr<int64_t> &&demo_prog_sum_out);
	CUDAContext(CUDAContext &&) = default;
	CUDAContext &operator=(CUDAContext &&) = default;
	CUDAContext(const CUDAContext &) = delete;
	CUDAContext &operator=(const CUDAContext &) = delete;

	std::vector<CUDAProgramRecord> cuda_progs;

	virtual ~CUDAContext();
	void set_module(CUmodule raw_ptr)
	{
		probe_module_container.emplace(raw_ptr, cuda_module_destroyer);
	}
	void set_demo_module(CUmodule raw_ptr)
	{
		demo_module_container.emplace(raw_ptr, cuda_module_destroyer);
	}
};

std::optional<std::unique_ptr<cuda::CUDAContext> > create_cuda_context();

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
	std::optional<std::shared_ptr<std::atomic<bool> > >
	start_cuda_prober(int id);
	int start_cuda_demo_program();
	std::unique_ptr<cuda::CUDAContext> cuda_ctx;
};

} // namespace bpftime

#endif
