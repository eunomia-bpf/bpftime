#include "frida_attach_entry.hpp"
#include "frida_attach_utils.hpp"
#include "frida_uprobe_attach_impl.hpp"
#include <variant>
using namespace bpftime::attach;
int frida_attach_entry::get_type() const
{
	if (std::holds_alternative<callback_variant>(callback)) {
		return from_cb_idx_to_attach_type(
			std::get<callback_variant>(callback).index());
	} else {
		return std::get<ebpf_callback_args>(callback).attach_type;
	}
}
