#include "bpftime_driver.hpp"
#include "ebpf-vm.h"
#include <spdlog/spdlog.h>
#include <json.hpp>

using namespace bpftime;
using json = nlohmann::json;
using namespace std;

static json bpf_map_attr_to_json(const bpf_map_attr &attr)
{
	json j;
	j["map_type"] = attr.type;
	j["key_size"] = attr.key_size;
	j["value_size"] = attr.value_size;
	j["max_entries"] = attr.max_ents;
	j["flags"] = attr.flags;
	j["ifindex"] = attr.ifindex;
	j["btf_vmlinux_value_type_id"] = attr.btf_vmlinux_value_type_id;
	j["btf_id"] = attr.btf_id;
	j["btf_key_type_id"] = attr.btf_key_type_id;
	j["btf_value_type_id"] = attr.btf_value_type_id;
	j["map_extra"] = attr.map_extra;
	return j;
}


