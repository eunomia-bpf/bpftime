#include "bpftime_shm.hpp"
#include "handler/epoll_handler.hpp"
#include "handler/perf_event_handler.hpp"
#include "spdlog/spdlog.h"
#include <bpftime_shm_internal.hpp>
#include <cstdio>
#include <sys/epoll.h>
#include <unistd.h>
#include <variant>
#include <fstream>
#include <json.hpp>
#include "bpftime_shm_json.hpp"

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

static bpf_map_attr json_to_bpf_map_attr(const json &j)
{
	bpf_map_attr attr;
	attr.type = j["map_type"];
	attr.key_size = j["key_size"];
	attr.value_size = j["value_size"];
	attr.max_ents = j["max_entries"];
	attr.flags = j["flags"];
	attr.ifindex = j["ifindex"];
	attr.btf_vmlinux_value_type_id = j["btf_vmlinux_value_type_id"];
	attr.btf_id = j["btf_id"];
	attr.btf_key_type_id = j["btf_key_type_id"];
	attr.btf_value_type_id = j["btf_value_type_id"];
	attr.map_extra = j["map_extra"];
	return attr;
}

static json
bpf_perf_event_handler_to_json(const bpf_perf_event_handler &handler)
{
	json j;
	j["type"] = handler.type;
	j["offset"] = handler.offset;
	j["pid"] = handler.pid;
	j["ref_ctr_off"] = handler.ref_ctr_off;
	j["_module_name"] = handler._module_name;
	j["tracepoint_id"] = handler.tracepoint_id;
	return j;
}

extern "C" int bpftime_import_global_shm_from_json(const char *filename)
{
	assert(false && "Not implemented");
	return -1;
}

int bpftime::bpftime_import_shm_from_json(const bpftime_shm &shm,
					  const char *filename)
{
	return -1;
}

extern "C" int bpftime_export_global_shm_to_json(const char *filename)
{
	return bpftime_export_shm_to_json(shm_holder.global_shared_memory, filename);
}

int bpftime::bpftime_export_shm_to_json(const bpftime_shm &shm,
					const char *filename)
{
	std::ofstream file(filename);
	json j;

	const handler_manager *manager = shm.get_manager();
	if (!manager) {
		spdlog::error("No manager found in the shared memory");
		return -1;
	}
	for (std::size_t i = 0; i < manager->size(); i++) {
		// skip uninitialized handlers
		if (!manager->is_allocated(i)) {
			continue;
		}
		auto &handler = manager->get_handler(i);
		// load the bpf prog
		if (std::holds_alternative<bpf_prog_handler>(handler)) {
			auto &prog_handler =
				std::get<bpf_prog_handler>(handler);
			const ebpf_inst *insns = prog_handler.insns.data();
			size_t cnt = prog_handler.insns.size();
			const char *name = prog_handler.name.c_str();
			// record the prog into json, key is the index of the
			// prog
			j[std::to_string(i)] = {
				{ "type", "bpf_prog_handler" },
				{ "insns", buffer_to_hex_string(
						   (const unsigned char *)insns,
						   sizeof(ebpf_inst) * cnt) },
				{ "cnt", cnt },
				{ "name", name }
			};
			spdlog::info("find prog fd={} name={}", i,
				     prog_handler.name);
		} else if (std::holds_alternative<bpf_map_handler>(handler)) {
			auto &map_handler = std::get<bpf_map_handler>(handler);
			const char *name = map_handler.name.c_str();
			spdlog::info("bpf_map_handler name={} found at {}",
				     name, i);
			bpf_map_attr attr = map_handler.attr;
			j[std::to_string(i)] = { { "type", "bpf_map_handler" },
						 { "name", name },
						 { "attr", bpf_map_attr_to_json(
								   attr) } };
		} else if (std::holds_alternative<bpf_perf_event_handler>(
				   handler)) {
			auto &perf_handler =
				std::get<bpf_perf_event_handler>(handler);
			j[std::to_string(i)] = {
				{ "type", "bpf_perf_event_handler" },
				{ "data",
				  bpf_perf_event_handler_to_json(perf_handler) }
			};
			spdlog::info("bpf_perf_event_handler found at {}", i);
		} else if (std::holds_alternative<epoll_handler>(handler)) {
			auto &h = std::get<epoll_handler>(handler);
			j[std::to_string(i)] = { { "type", "epoll_handler" } };
			spdlog::info("epoll_handler found at {}", i);
		} else if (std::holds_alternative<bpf_link_handler>(handler)) {
			auto &h = std::get<bpf_link_handler>(handler);
			j[std::to_string(i)] = { { "type", "bpf_link_handler" },
						 { "prog_fd", h.prog_fd },
						 { "target_fd", h.target_fd } };
			spdlog::info(
				"bpf_link_handler found at {}，link {} -> {}",
				i, h.prog_fd, h.target_fd);
		} else {
			spdlog::error("Unsupported handler type {}",
				      handler.index());
			return -1;
		}
	}
	// write the json to file
	file << j.dump(4);
	file.close();
	return 0;
}
