#ifndef _BPF_ATTACH_CTX
#define _BPF_ATTACH_CTX

#include <cinttypes>
#include "bpftime_config.hpp"
#include <map>
#include <vector>
#include <memory>

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

	// attach to a function in the object. the bpf program will be called
	// before the
	// function execution.
	// int create_uprobe(void *function, int id, bool retprobe = false);
	// filter the function execution.
	// int create_filter(void *function);
	// int create_filter(void *function, int id);
	// hook a function to new_function
	// int create_replace_with_handler(int id, bpftime_hook_entry_type type,
	// 				void *function, void *handler_func);
	// the bpf program will be called instead of the function execution.
	// int create_replace(void *function);
	// create a replace function with an id
	// int create_replace(void *function, int id);
	// Create a syscall tracepoint, recording its corresponding program into
	// syscall_entry_progs and syscall_exit_progs
	int create_tracepoint(int tracepoint_id, int perf_fd,
			      const handler_manager *manager);
	// int destory_attach(int id);

	// attach prog to a given attach id
	// int attach_prog(const bpftime_prog *prog, int id);
	// the bpf program will be called instead of the function execution.
	// int detach(const bpftime_prog *prog);

	// replace the function for the old program. prog can be nullptr
	// int replace_func(void *new_function, void *target_function, void
	// *data); revert or recover the function for the old program int
	// revert_func(void *target_function);

	// create bpf_attach_ctx from handler_manager in shared memory
	int init_attach_ctx_from_handlers(const handler_manager *manager,
					  const agent_config &config);
	// create bpf_attach_ctx from handler_manager in global_shared_memory
	int init_attach_ctx_from_handlers(const agent_config &config);

	// // find the function by name in current process
	// // must be called after init attach_ctx
	// void *find_function_by_name(const char *name);
	// // find module export function by name
	// // must be called after init attach_ctx
	// void *module_find_export_by_name(const char *module_name,
	// 				 const char *symbol_name);
	// // get the base addr of a module
	// // must be called after init attach_ctx
	// void *module_get_base_addr(const char *module_name);

	// Check whether there is a syscall trace program. Use the global
	// handler manager
	bool check_exist_syscall_trace_program();
	// Check whether there is a syscall trace program
	bool check_exist_syscall_trace_program(const handler_manager *manager);

	// Check whether a certain pid was already equipped with syscall tracer
	// Using a set stored in the shared memory
	bool check_syscall_trace_setup(int pid);
	// Set whether a certain pid was already equipped with syscall tracer
	// Using a set stored in the shared memory
	void set_syscall_trace_setup(int pid, bool whether);

	int64_t run_syscall_hooker(int64_t sys_nr, int64_t arg1, int64_t arg2,
				   int64_t arg3, int64_t arg4, int64_t arg5,
				   int64_t arg6);
	void set_orig_syscall_func(syscall_hooker_func_t f)
	{
		orig_syscall = f;
	}
	base_attach_manager &get_attach_manager()
	{
		return *attach_manager;
	}

    private:
	// add uprobe listener
	// int add_listener(GumInvocationListener *listener, void
	// *target_function, 		 void *data);
	constexpr static int CURRENT_ID_OFFSET = 65536;
	volatile int current_id = CURRENT_ID_OFFSET;
	// frida gum interceptor
	// GumInterceptor *interceptor = nullptr;
	// map between function and bpf program
	// std::map<void *, hook_entry> hook_entry_table;
	// map between fd and function
	// std::map<int, void *> hook_entry_index;

	// save the progs for memory management
	std::map<int, std::unique_ptr<bpftime_prog> > progs;

	std::vector<const bpftime_prog *> sys_enter_progs[512];
	std::vector<const bpftime_prog *> sys_exit_progs[512];
	std::vector<const bpftime_prog *> global_sys_enter_progs;
	std::vector<const bpftime_prog *> global_sys_exit_progs;

	syscall_hooker_func_t orig_syscall = nullptr;
	std::unique_ptr<base_attach_manager> attach_manager;
};

} // namespace bpftime

#endif
