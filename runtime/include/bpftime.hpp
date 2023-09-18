#ifndef _BPFTIME_RUNTIME_HEADERS_H_
#define _BPFTIME_RUNTIME_HEADERS_H_

#include <memory>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <cstdint>
#include <cassert>
#include <cstdlib>
#include <ebpf-core.h>

extern "C" {

struct trace_entry {
	short unsigned int type;
	unsigned char flags;
	unsigned char preempt_count;
	int pid;
};

struct trace_event_raw_sys_enter {
	struct trace_entry ent;
	long int id;
	long unsigned int args[6];
	char __data[0];
};
struct trace_event_raw_sys_exit {
	struct trace_entry ent;
	long int id;
	long int ret;
	char __data[0];
};
struct _FridaUprobeListener;
typedef struct _GumInterceptor GumInterceptor;
typedef struct _GumInvocationListener GumInvocationListener;
}

namespace bpftime
{
using syscall_hooker_func_t = int64_t (*)(int64_t sys_nr, int64_t arg1,
					  int64_t arg2, int64_t arg3,
					  int64_t arg4, int64_t arg5,
					  int64_t arg6);

struct bpftime_helper_info;

// executable program for bpf function
class bpftime_prog {
    public:
	const char *prog_name() const
	{
		return name.c_str();
	}
	bpftime_prog(const ebpf_inst *insn, size_t insn_cnt, const char *name);
	~bpftime_prog();

	// load the programs to userspace vm or compile the jit program
	// if program_name is NULL, will load the first program in the object
	int bpftime_prog_load(bool jit);
	int bpftime_prog_unload();

	// exec in user space
	int bpftime_prog_exec(void *memory, size_t memory_size,
			      uint64_t *return_val) const;
	int bpftime_prog_register_raw_helper(struct bpftime_helper_info info);
	const std::vector<ebpf_inst> &get_insns() const
	{
		return insns;
	}

    private:
	int bpftime_prog_set_insn(struct ebpf_inst *insn, size_t insn_cnt);
	std::string name;
	// vm at the first element
	struct ebpf_vm *vm;

	bool jitted;
	// used in jit
	ebpf_jit_fn fn;
	std::vector<struct ebpf_inst> insns;

	char *errmsg;

	// ffi ctx
	struct bpftime_ffi_ctx *ffi_ctx;
};

// helper context for bpf execution
class bpftime_helper_group;

struct bpftime_helper_info {
	unsigned int index;
	std::string name;
	void *fn;
};

class bpftime_helper_group {
    public:
	bpftime_helper_group() = default;
	bpftime_helper_group(
		std::map<unsigned int, bpftime_helper_info> helper_map)
		: helper_map(helper_map)
	{
	}
	~bpftime_helper_group() = default;

	// Register a helper
	int register_helper(const bpftime_helper_info &info);

	// Append another group to the current one
	int append(const bpftime_helper_group &another_group);

	// Utility function to get the FFI helper group
	static const bpftime_helper_group &get_ffi_helper_group();

	// Utility function to get the kernel utilities helper group
	static const bpftime_helper_group &get_kernel_utils_helper_group();

	// Function to register and create a local hash map helper group
	static const bpftime_helper_group &get_shm_maps_helper_group();

	// Add the helper group to the program
	int add_helper_group_to_prog(bpftime_prog *prog) const;

    private:
	// Map to store helpers indexed by their unique ID
	std::map<unsigned int, bpftime_helper_info> helper_map;
};

#define MAX_FFI_FUNCS 8192 * 4
#define MAX_ARGS_COUNT 6
#define MAX_FUNC_NAME_LEN 64

class bpftime_prog;
class bpftime_helper_group;
struct bpftime_ffi_ctx;
class bpf_attach_ctx;

enum ffi_types {
	FFI_TYPE_UNKNOWN,
	FFI_TYPE_VOID,
	FFI_TYPE_INT8,
	FFI_TYPE_UINT8,
	FFI_TYPE_INT16,
	FFI_TYPE_UINT16,
	FFI_TYPE_INT32,
	FFI_TYPE_UINT32,
	FFI_TYPE_INT64,
	FFI_TYPE_UINT64,
	FFI_TYPE_FLOAT,
	FFI_TYPE_DOUBLE,
	FFI_TYPE_POINTER,
	FFI_TYPE_STRUCT,
};

typedef void *(*ffi_func)(void *r1, void *r2, void *r3, void *r4, void *r5);

/* Useful for eliminating compiler warnings.  */
#define FFI_FN(f) ((ffi_func)(void *)((void (*)(void))f))

struct ebpf_ffi_func_info {
	char name[MAX_FUNC_NAME_LEN];
	ffi_func func;
	enum ffi_types ret_type;
	enum ffi_types arg_types[MAX_ARGS_COUNT];
	int num_args;
	int id;
	bool is_attached;
};

struct arg_list {
	uint64_t args[MAX_ARGS_COUNT];
};

union arg_val {
	uint64_t uint64;
	int64_t int64;
	double double_val;
	void *ptr;
};

union arg_val to_arg_val(enum ffi_types type, uint64_t val);

uint64_t from_arg_val(enum ffi_types type, union arg_val val);

// register a ffi for a program
void bpftime_ffi_register_ffi(uint64_t id, ebpf_ffi_func_info func_info);

// register a ffi for a program base on info.
// probe ctx will find the function address and fill in the func_info
int bpftime_ffi_resolve_from_info(bpf_attach_ctx *probe_ctx,
				  ebpf_ffi_func_info func_info);

#ifndef MAX_BPF_PROG
#define MAX_BPF_PROG 4096
#endif

enum bpftime_hook_entry_type {
	BPFTIME_UNSPEC = 0,
	BPFTIME_REPLACE = 1,
	BPFTIME_UPROBE = 2,
	BPFTIME_SYSCALL = 3,
	__MAX_BPFTIME_ATTACH_TYPE = 4,
};

enum PatchOp {
	OP_SKIP,
	OP_RESUME,
};

struct hook_entry {
	bpftime_hook_entry_type type = BPFTIME_UNSPEC;

	int id = -1;
	// the function to be hooked
	void *hook_func = nullptr;
	// the bpf program
	std::set<const bpftime_prog *> progs;

	// the data for the bpf program
	void *data = nullptr;
	void *ret_val = nullptr;

	// filter or replace
	void *handler_function = nullptr;

	// listener for uprobe
	GumInvocationListener *listener = nullptr;
	int uretprobe_id;
	std::set<const bpftime_prog *> ret_progs;
};

struct agent_config {
	bool debug = false;
	bool jit_enabled = false;

	// helper groups
	bool enable_kernel_helper_group = true;
	bool enable_ffi_helper_group = false;
	bool enable_shm_maps_helper_group = true;
};
class handler_manager;

class bpf_attach_ctx {
    public:
	bpf_attach_ctx();
	~bpf_attach_ctx();

	// attach to a function in the object. the bpf program will be called
	// before the
	// function execution.
	int create_uprobe(void *function, int id, bool retprobe = false);
	// filter the function execution.
	int create_filter(void *function);
	int create_filter(void *function, int id);
	// hook a function to new_function
	int create_replace_with_handler(int id, bpftime_hook_entry_type type,
					void *function, void *handler_func);
	// the bpf program will be called instead of the function execution.
	int create_replace(void *function);
	// create a replace function with an id
	int create_replace(void *function, int id);
	// Create a syscall tracepoint, recording its corresponding program into
	// syscall_entry_progs and syscall_exit_progs
	int create_tracepoint(int tracepoint_id, int perf_fd,
			      const handler_manager *manager);
	int destory_attach(int id);

	// attach prog to a given attach id
	int attach_prog(const bpftime_prog *prog, int id);
	// the bpf program will be called instead of the function execution.
	int detach(const bpftime_prog *prog);

	// replace the function for the old program. prog can be nullptr
	int replace_func(void *new_function, void *target_function, void *data);
	// revert or recover the function for the old program
	int revert_func(void *target_function);

	// create bpf_attach_ctx from handler_manager in shared memory
	int init_attach_ctx_from_handlers(const handler_manager *manager,
					  agent_config &config);
	// create bpf_attach_ctx from handler_manager in global_shared_memory
	int init_attach_ctx_from_handlers(agent_config &config);
	// attach progs with fds to the fds in manager
	int attach_progs_in_manager(const handler_manager *manager);

	// find the function by name in current process
	// must be called after init attach_ctx
	void *find_function_by_name(const char *name);
	// find module export function by name
	// must be called after init attach_ctx
	void *module_find_export_by_name(const char *module_name,
					 const char *symbol_name);
	// get the base addr of a module
	// must be called after init attach_ctx
	void *module_get_base_addr(const char *module_name);

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

    private:
	// add uprobe listener
	int add_listener(GumInvocationListener *listener, void *target_function,
			 void *data);
	constexpr static int CURRENT_ID_OFFSET = 65536;
	volatile int current_id = CURRENT_ID_OFFSET;
	// frida gum interceptor
	GumInterceptor *interceptor = nullptr;
	// map between function and bpf program
	std::map<void *, hook_entry> hook_entry_table;
	// map between fd and function
	std::map<int, void *> hook_entry_index;

	// save the progs for memory management
	std::map<int, std::unique_ptr<bpftime_prog> > progs;

	std::vector<const bpftime_prog *> sys_enter_progs[512];
	std::vector<const bpftime_prog *> sys_exit_progs[512];
	syscall_hooker_func_t orig_syscall = nullptr;
};
// hook entry is store in frida context or other context.
// You can get the hook entry from context. for example:
// gum_invocation_context_get_replacement_data(ctx);
//
// hook_entry is only valid in th hooked function.
struct hook_entry;
class bpftime_prog;

// get hook entry from probe context
const hook_entry *bpftime_probe_get_hook_entry(void);
// get prog from hook entry
const bpftime_prog *
bpftime_probe_get_prog_from_hook_entry(const hook_entry *hook_entry);

} // namespace bpftime

#endif
