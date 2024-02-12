#include "frida_attach_entry.hpp"
using namespace bpftime::attach;
int frida_attach_entry::get_type() const
{
	return (int)cb.index();
}
