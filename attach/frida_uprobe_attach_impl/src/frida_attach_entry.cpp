#include "frida_attach_entry.hpp"
#include "frida_attach_utils.hpp"
#include "frida_uprobe_attach_impl.hpp"
using namespace bpftime::attach;
int frida_attach_entry::get_type() const
{
	return from_cb_idx_to_attach_type(cb.index());
}
