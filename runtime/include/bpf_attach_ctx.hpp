#ifndef _BPF_ATTACH_CTX
#define _BPF_ATTACH_CTX

#include "bpftime_config.hpp"
#include <map>
#include <memory>
#include <frida_uprobe_attach_impl.hpp>
#include <set>
#include <syscall_trace_attach_impl.hpp>
typedef struct _GumInterceptor GumInterceptor;
typedef struct _GumInvocationListener GumInvocationListener;

namespace bpftime
{
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

	attach::base_attach_impl &get_uprobe_attach_impl()
	{
		return *frida_uprobe_attach_impl;
	}

	attach::base_attach_impl &get_syscall_attach_impl()
	{
		return *syscall_trace_attach_impl;
	}

    private:
	constexpr static int CURRENT_ID_OFFSET = 65536;
	volatile int current_id = CURRENT_ID_OFFSET;

	// save the progs for memory management
	std::map<int, std::unique_ptr<bpftime_prog> > progs;
	std::map<int, int> attach_ids;
	// std::unique_ptr<attach::base_attach_impl> frida_uprobe_attach_impl;
	// std::unique_ptr<attach::base_attach_impl> syscall_trace_attach_impl;

	std::set<int> instantiated_handlers;

	int instantiate_handler_at(const handler_manager *manager, int id,
				   std::set<int> &stk,
				   const agent_config &config);
};

} // namespace bpftime

#endif
