#ifndef _HOOK_ENTRY_HPP
#define _HOOK_ENTRY_HPP
#include <set>
#include "bpftime_prog.hpp"
typedef struct _GumInvocationListener GumInvocationListener;
namespace bpftime
{
enum bpftime_hook_entry_type {
	BPFTIME_UNSPEC = 0,
	BPFTIME_REPLACE = 1,
	BPFTIME_UPROBE = 2,
	BPFTIME_SYSCALL = 3,
	__MAX_BPFTIME_ATTACH_TYPE = 4,
};
// hook entry is store in frida context or other context.
// You can get the hook entry from context. for example:
// gum_invocation_context_get_replacement_data(ctx);
//
// hook_entry is only valid in th hooked function.
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
// get hook entry from probe context
const hook_entry *bpftime_probe_get_hook_entry(void);
// get prog from hook entry
const bpftime_prog *
bpftime_probe_get_prog_from_hook_entry(const hook_entry *hook_entry);

} // namespace bpftime

#endif
