/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpftime_shm.hpp"
#include <ebpf-vm.h>
#include "handler/epoll_handler.hpp"
#include "handler/perf_event_handler.hpp"
#include "spdlog/spdlog.h"
#include <bpftime_shm_internal.hpp>
#include <cstdio>
#if __linux__
#include <sys/epoll.h>
#elif __APPLE__
#include "bpftime_epoll.h"
#endif
#include <unistd.h>
#include <algorithm>
#include <utility>
#include <vector>
#include <variant>
#include <fstream>
#include <json.hpp>
#include "bpftime_shm_json.hpp"
#ifdef ENABLE_BPFTIME_VERIFIER
#include <bpftime-verifier.hpp>
#endif
#include <cerrno>
#include <charconv>
#include <climits>
#include <limits>
#include <set>
#include <stdexcept>
#include <system_error>
#include <type_traits>

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

	j["kernel_bpf_map_id"] = attr.kernel_bpf_map_id;
	return j;
}

static json
bpf_perf_event_handler_attr_to_json(const bpf_perf_event_handler &handler)
{
	json j;
	j["type"] = handler.type;
	if (std::holds_alternative<uprobe_perf_event_data>(handler.data)) {
		auto &data = std::get<uprobe_perf_event_data>(handler.data);
		j["data_type"] = "uprobe_perf_event_data";
		j["offset"] = data.offset;
		j["pid"] = data.pid;
		j["ref_ctr_off"] = data.ref_ctr_off;
		j["_module_name"] = data._module_name;
	} else if (std::holds_alternative<tracepoint_perf_event_data>(
			   handler.data)) {
		auto &data = std::get<tracepoint_perf_event_data>(handler.data);
		j["data_type"] = "tracepoint_perf_event_data";
		j["pid"] = data.pid;
		j["tracepoint_id"] = data.tracepoint_id;
	} else if (std::holds_alternative<software_perf_event_shared_ptr>(
			   handler.data)) {
		j["data_type"] = "software_perf_event_shared_ptr";
		auto &data =
			std::get<software_perf_event_shared_ptr>(handler.data);
		j["cpu"] = data->cpu;
		j["config"] = data->config;
		j["sample_type"] = data->sample_type;
	}

	return j;
}

namespace
{
// JSON is a portable definition graph, not a live-memory checkpoint.
// Callers must quiesce the registry until the import completes.
[[noreturn]] void invalid(const std::string &message)
{
	throw std::invalid_argument(message);
}

template <typename T> T integer(const json &value)
{
	static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>);
	if (!value.is_number_integer())
		invalid("expected an integer");
	if (value.is_number_unsigned()) {
		auto n = value.get<uint64_t>();
		if (n > static_cast<uint64_t>(std::numeric_limits<T>::max()))
			invalid("integer is out of range");
		return static_cast<T>(n);
	}
	auto n = value.get<int64_t>();
	if constexpr (std::is_unsigned_v<T>) {
		if (n < 0 || static_cast<uint64_t>(n) >
				     std::numeric_limits<T>::max())
			invalid("integer is out of range");
	} else if (n < std::numeric_limits<T>::min() ||
		   n > std::numeric_limits<T>::max()) {
		invalid("integer is out of range");
	}
	return static_cast<T>(n);
}

template <typename T>
T number(const json &object, const char *key, T default_value)
{
	return object.contains(key) ? integer<T>(object.at(key)) : default_value;
}

std::string string_field(const json &object, const char *key)
{
	auto value = object.at(key).get<std::string>();
	if (value.find('\0') != std::string::npos)
		invalid(std::string(key) + " contains a NUL byte");
	return value;
}

bool boolean(const json &value)
{
	if (!value.is_boolean())
		invalid("expected a boolean");
	return value.get<bool>();
}

int descriptor(const std::string &key)
{
	if (key.empty() || (key.size() > 1 && key[0] == '0') ||
	    !std::all_of(key.begin(), key.end(),
			 [](char c) { return c >= '0' && c <= '9'; }))
		invalid("handler key must be a canonical nonnegative decimal fd");
	int fd = -1;
	auto result = std::from_chars(key.data(), key.data() + key.size(), fd);
	if (result.ec != std::errc{} || result.ptr != key.data() + key.size())
		invalid("handler fd is out of range");
	if (fd <= STDERR_FILENO)
		invalid("handler fd is reserved for standard streams: " + key);
	return fd;
}

template <typename Input> json parse_document(Input &&input)
{
	std::vector<std::set<std::string>> object_keys;
	return json::parse(std::forward<Input>(input),
			   [&](int depth, json::parse_event_t event, json &value) {
				   if (depth > 64) invalid("JSON nesting is too deep");
				   if (event == json::parse_event_t::object_start)
					   object_keys.emplace_back();
				   else if (event == json::parse_event_t::key) {
					   if (!object_keys.back()
							.insert(value.get<std::string>())
							.second)
						   invalid("duplicate JSON object key");
				   } else if (event == json::parse_event_t::object_end)
					   object_keys.pop_back();
				   return true;
			   });
}

struct map_definition {
	std::string name;
	bpf_map_attr attr;
};
struct program_definition {
	std::string name;
	int type;
	std::vector<ebpf_inst> insns;
	std::vector<int> legacy_targets;
};
struct perf_definition {
	int type;
	bool enabled;
	std::string name;
	uint64_t offset = 0;
	size_t ref_ctr_off = 0;
	int pid = 0;
	int cpu = 0;
	int32_t tracepoint_id = 0;
	int32_t sample_type = 0;
	int64_t config = 0;
};
struct link_definition {
	int prog_fd;
	int target_fd;
};
using definition = std::variant<map_definition, program_definition,
				perf_definition, link_definition>;
using import_plan = std::map<int, definition>;

// Only implementations owned by the segment can be staged here. Restoration
// of GPU, kernel-backed and custom maps is a separate compatibility decision.
void validate_import_map(const bpf_map_attr &attr)
{
	switch (static_cast<bpf_map_type>(attr.type)) {
	case bpf_map_type::BPF_MAP_TYPE_HASH:
	case bpf_map_type::BPF_MAP_TYPE_ARRAY:
	case bpf_map_type::BPF_MAP_TYPE_PERF_EVENT_ARRAY:
	case bpf_map_type::BPF_MAP_TYPE_PERCPU_HASH:
	case bpf_map_type::BPF_MAP_TYPE_PERCPU_ARRAY:
	case bpf_map_type::BPF_MAP_TYPE_LRU_HASH:
	case bpf_map_type::BPF_MAP_TYPE_LPM_TRIE:
	case bpf_map_type::BPF_MAP_TYPE_ARRAY_OF_MAPS:
	case bpf_map_type::BPF_MAP_TYPE_QUEUE:
	case bpf_map_type::BPF_MAP_TYPE_STACK:
	case bpf_map_type::BPF_MAP_TYPE_RINGBUF:
	case bpf_map_type::BPF_MAP_TYPE_BLOOM_FILTER:
		break;
#ifdef BPFTIME_BUILD_WITH_LIBBPF
	case bpf_map_type::BPF_MAP_TYPE_PROG_ARRAY:
	case bpf_map_type::BPF_MAP_TYPE_STACK_TRACE:
		break;
#endif
	default:
		invalid("map type cannot be restored transactionally by this build: " +
			std::to_string(attr.type));
	}
	const auto type = static_cast<bpf_map_type>(attr.type);
	if (attr.max_ents == 0)
		invalid("map max_entries must be positive");
	if (type == bpf_map_type::BPF_MAP_TYPE_RINGBUF) {
		if (attr.key_size || attr.value_size ||
		    (attr.max_ents & (attr.max_ents - 1)))
			invalid("ring buffer requires zero key/value sizes and power-of-two capacity");
		const auto page_size = sysconf(_SC_PAGESIZE);
		if (page_size <= 0 ||
		    2 * static_cast<uint64_t>(page_size) + 2 * static_cast<uint64_t>(attr.max_ents) > UINT32_MAX)
			invalid("ring buffer dimensions overflow implementation size arithmetic");
		return;
	}
	if (attr.value_size == 0)
		invalid("map value_size must be positive");
	const bool no_key = type == bpf_map_type::BPF_MAP_TYPE_QUEUE ||
			    type == bpf_map_type::BPF_MAP_TYPE_STACK ||
			    type == bpf_map_type::BPF_MAP_TYPE_BLOOM_FILTER;
	if ((no_key && attr.key_size != 0) || (!no_key && attr.key_size == 0))
		invalid("invalid map key_size");
	if ((type == bpf_map_type::BPF_MAP_TYPE_ARRAY ||
	     type == bpf_map_type::BPF_MAP_TYPE_PERCPU_ARRAY ||
	     type == bpf_map_type::BPF_MAP_TYPE_PERF_EVENT_ARRAY ||
	     type == bpf_map_type::BPF_MAP_TYPE_ARRAY_OF_MAPS ||
	     type == bpf_map_type::BPF_MAP_TYPE_PROG_ARRAY) &&
	    attr.key_size != sizeof(uint32_t))
		invalid("array map key_size must be four");
	if ((type == bpf_map_type::BPF_MAP_TYPE_ARRAY_OF_MAPS ||
	     type == bpf_map_type::BPF_MAP_TYPE_PROG_ARRAY ||
	     type == bpf_map_type::BPF_MAP_TYPE_PERF_EVENT_ARRAY) &&
	    attr.value_size != sizeof(uint32_t))
		invalid("fd array map value_size must be four");
	if (type == bpf_map_type::BPF_MAP_TYPE_LPM_TRIE &&
	    (attr.key_size <= sizeof(uint32_t) ||
	     attr.key_size > sizeof(uint32_t) + 256))
		invalid("LPM trie key must include a prefix and 1..256 address bytes");
	if (type == bpf_map_type::BPF_MAP_TYPE_STACK_TRACE &&
	    (attr.key_size != sizeof(uint32_t) || attr.value_size % sizeof(uint64_t)))
		invalid("stack trace map requires a four-byte key and eight-byte value stride");

	// Check the arithmetic actually used by each implementation, not a
	// hypothetical dense key/value allocation for every map. Bloom filters,
	// sparse maps, and lazy kernel wrappers do not allocate max_entries values.
	uint64_t cpu_count = 1;
	if (type == bpf_map_type::BPF_MAP_TYPE_PERCPU_ARRAY ||
	    type == bpf_map_type::BPF_MAP_TYPE_PERCPU_HASH) {
		auto count = sysconf(_SC_NPROCESSORS_ONLN);
		if (count <= 0 || static_cast<uint64_t>(count) > INT_MAX)
			invalid("cannot determine per-CPU map size");
		cpu_count = static_cast<uint64_t>(count);
	}
	auto checked_product = [](uint64_t count, uint64_t stride, uint64_t limit) {
		if (stride > limit || count > limit / stride)
			invalid("map dimensions overflow implementation size arithmetic");
	};
	switch (type) {
	case bpf_map_type::BPF_MAP_TYPE_ARRAY:
	case bpf_map_type::BPF_MAP_TYPE_ARRAY_OF_MAPS:
	case bpf_map_type::BPF_MAP_TYPE_QUEUE:
	case bpf_map_type::BPF_MAP_TYPE_STACK:
		checked_product(attr.max_ents, attr.value_size, UINT32_MAX);
		break;
	case bpf_map_type::BPF_MAP_TYPE_PERCPU_ARRAY:
		checked_product(cpu_count, attr.value_size, UINT32_MAX);
		checked_product(attr.max_ents, cpu_count * attr.value_size, UINT32_MAX);
		break;
	case bpf_map_type::BPF_MAP_TYPE_PERCPU_HASH:
		checked_product(cpu_count, attr.value_size, UINT32_MAX);
		break;
	case bpf_map_type::BPF_MAP_TYPE_HASH:
#ifndef BPFTIME_USE_VAR_HASH_MAP
		// fix_size_hash_map_impl stores the rounded bucket count as uint32_t.
		// The largest prime it can represent is 2^32 - 5.
		if (attr.max_ents > UINT32_MAX - 4)
			invalid("fixed hash rounded bucket count exceeds uint32_t");
		// The fixed hash rounds its bucket count up to a prime. Twice the
		// requested count bounds that rounding without repeating its search.
		checked_product(2 * static_cast<uint64_t>(attr.max_ents),
				4 + static_cast<uint64_t>(attr.key_size) + attr.value_size,
				std::numeric_limits<size_t>::max());
#endif
		break;
	case bpf_map_type::BPF_MAP_TYPE_BLOOM_FILTER: {
		const uint64_t hashes = (attr.map_extra & 0xf) ? (attr.map_extra & 0xf) : 5;
		checked_product(attr.max_ents, hashes * 7, std::numeric_limits<size_t>::max());
		break;
	}
	default:
		break; // The remaining constructors use checked containers or lazy state.
	}
}

bpf_map_attr decode_map(const json &j)
{
	bpf_map_attr a;
	a.type = integer<int>(j.at("map_type"));
	a.key_size = integer<uint32_t>(j.at("key_size"));
	a.value_size = integer<uint32_t>(j.at("value_size"));
	a.max_ents = integer<uint32_t>(j.at("max_entries"));
	a.flags = number<uint64_t>(j, "flags", 0);
	a.ifindex = number<uint32_t>(j, "ifindex", 0);
	a.btf_vmlinux_value_type_id = number<uint32_t>(j, "btf_vmlinux_value_type_id", 0);
	a.btf_id = number<uint32_t>(j, "btf_id", 0);
	a.btf_key_type_id = number<uint32_t>(j, "btf_key_type_id", 0);
	a.btf_value_type_id = number<uint32_t>(j, "btf_value_type_id", 0);
	a.map_extra = number<uint64_t>(j, "map_extra", 0);
	a.kernel_bpf_map_id = number<uint32_t>(j, "kernel_bpf_map_id", 0);
	return a;
}

perf_definition decode_perf(const json &value)
{
	const auto &a = value.at("attr");
	perf_definition p{};
	p.type = integer<int>(a.at("type"));
	p.enabled = boolean(value.at("enabled"));
	const char *expected_data_type;
	switch (static_cast<bpf_event_type>(p.type)) {
	case bpf_event_type::BPF_TYPE_UPROBE:
	case bpf_event_type::BPF_TYPE_URETPROBE:
	case bpf_event_type::BPF_TYPE_UPROBE_OVERRIDE:
		expected_data_type = "uprobe_perf_event_data";
		p.pid = integer<int>(a.at("pid"));
		p.offset = integer<uint64_t>(a.at("offset"));
		p.ref_ctr_off = number<size_t>(a, "ref_ctr_off", 0);
		p.name = string_field(a, "_module_name");
		break;
	case bpf_event_type::PERF_TYPE_TRACEPOINT:
		expected_data_type = "tracepoint_perf_event_data";
		p.pid = integer<int>(a.at("pid"));
		p.tracepoint_id = integer<int32_t>(a.at("tracepoint_id"));
		break;
	case bpf_event_type::PERF_TYPE_SOFTWARE:
		expected_data_type = "software_perf_event_shared_ptr";
		p.cpu = integer<int>(a.at("cpu"));
		p.sample_type = integer<int32_t>(a.at("sample_type"));
		p.config = integer<int64_t>(a.at("config"));
		break;
	default:
		invalid("unsupported perf event type");
	}
	if (a.contains("data_type") &&
	    string_field(a, "data_type") != expected_data_type)
		invalid("perf event type and data_type disagree");
	return p;
}

definition decode_handler(const json &value)
{
	const auto type = string_field(value, "type");
	if (type == "epoll_handler" || type == "memfd_handler")
		invalid("handler metadata is export-only: " + type);
	const auto &a = value.at("attr");
	if (!a.is_object())
		invalid("handler attr must be an object");
	if (type == "bpf_map_handler") {
		auto attr = decode_map(a);
		auto name = string_field(value, "name");
		validate_import_map(attr);
		return map_definition{ std::move(name), attr };
	}
	if (type == "bpf_prog_handler") {
		program_definition p;
		p.name = string_field(value, "name");
		p.type = integer<int>(a.at("type"));
		if (p.type < 0)
			invalid("program type must be nonnegative");
		const auto hex = string_field(a, "insns");
		const auto count = integer<size_t>(a.at("cnt"));
		if (hex.size() % (2 * sizeof(ebpf_inst)) ||
		    count != hex.size() / (2 * sizeof(ebpf_inst)))
			invalid("instruction count and bytecode length disagree");
		p.insns.resize(count);
		if (hex_string_to_buffer(hex, reinterpret_cast<unsigned char *>(p.insns.data()),
					 count * sizeof(ebpf_inst)) < 0)
			invalid("invalid instruction hex");
		if (a.contains("attach_fds")) {
			if (!a.at("attach_fds").is_array())
				invalid("attach_fds must be an array");
			for (const auto &fd : a.at("attach_fds")) {
				int target = integer<int>(fd);
				if (target < 0) invalid("negative attach fd");
				p.legacy_targets.push_back(target);
			}
		}
		return p;
	}
	if (type == "bpf_perf_event_handler")
		return decode_perf(value);
	if (type == "bpf_link_handler")
		return link_definition{ integer<int>(a.at("prog_fd")),
					integer<int>(a.at("target_fd")) };
	invalid("unsupported handler type: " + type);
}

template <typename IsType>
void validate_link_endpoints(int fd, int program, int target, IsType &&is_type)
{
	if (!is_type(program, true) || !is_type(target, false))
		invalid("link " + std::to_string(fd) +
			" must reference a program and a perf event");
}

import_plan decode_document(const json &document, const handler_manager &manager)
{
	import_plan plan;
	if (document.is_null())
		return plan; // Legacy empty exports used null rather than {}.
	if (!document.is_object())
		invalid("JSON root must be an fd-keyed object");
	for (const auto &[key, value] : document.items()) {
		const int fd = descriptor(key);
		if (static_cast<size_t>(fd) >= manager.size())
			invalid("handler fd exceeds registry capacity: " + key);
		if (manager.is_allocated(fd))
			invalid("handler fd is already occupied: " + key);
		try {
			plan.emplace(fd, decode_handler(value));
		} catch (const std::exception &e) {
			invalid("handler " + key + ": " + e.what());
		}
	}
	// Reserve explicit ids before translating the old program attach_fds
	// representation into links. Do not allocate OS fds or duplicate an
	// explicit link that already represents the same attachment.
	std::set<std::pair<int, int>> links;
	for (const auto &[fd, value] : plan)
		if (const auto *l = std::get_if<link_definition>(&value))
			links.emplace(l->prog_fd, l->target_fd);
	std::vector<std::pair<int, int>> legacy_links;
	for (const auto &[fd, value] : plan)
		if (const auto *p = std::get_if<program_definition>(&value))
			for (int target : p->legacy_targets)
				if (links.emplace(fd, target).second)
					legacy_links.emplace_back(fd, target);
	// Avoid standard handler slots. These are shared IDs, not OS-fd
	// reservations; unlike open_fake_fd(), this cannot prevent collisions
	// with ordinary host descriptors in this or another process.
	int next = STDERR_FILENO + 1;
	for (auto [prog, target] : legacy_links) {
		while (static_cast<size_t>(next) < manager.size() &&
		       (plan.count(next) || manager.is_allocated(next)))
			++next;
		if (static_cast<size_t>(next) >= manager.size())
			invalid("no free handler slot for legacy attachment");
		plan.emplace(next++, link_definition{ prog, target });
	}
	auto is_type = [&](int fd, bool program) {
		if (fd <= STDERR_FILENO || static_cast<size_t>(fd) >= manager.size())
			return false;
		if (auto it = plan.find(fd); it != plan.end())
			return program ? std::holds_alternative<program_definition>(it->second) :
					 std::holds_alternative<perf_definition>(it->second);
		const auto &existing = manager.get_handler(fd);
		return program ? std::holds_alternative<bpf_prog_handler>(existing) :
				 std::holds_alternative<bpf_perf_event_handler>(existing);
	};
	for (const auto &[fd, value] : plan)
		if (const auto *l = std::get_if<link_definition>(&value))
			validate_link_endpoints(fd, l->prog_fd, l->target_fd, is_type);
	return plan;
}

[[noreturn]] void io_error(const char *operation)
{
	throw std::system_error(errno, std::generic_category(), operation);
}

template <typename Function>
int json_operation(const char *operation, const char *target, Function &&run)
{
	int error = EIO;
	try {
		run();
		return 0;
	} catch (const std::system_error &e) {
		error = e.code().value();
		try { SPDLOG_ERROR("JSON {} {} failed: {}", operation,
				   target ? target : "(null)", e.what()); } catch (...) {}
	} catch (const boost::interprocess::bad_alloc &e) {
		error = ENOMEM;
		try { SPDLOG_ERROR("JSON {} {} failed: {}", operation,
				   target ? target : "(null)", e.what()); } catch (...) {}
	} catch (const std::bad_alloc &e) {
		error = ENOMEM;
		try { SPDLOG_ERROR("JSON {} {} failed: {}", operation,
				   target ? target : "(null)", e.what()); } catch (...) {}
	} catch (const std::exception &e) {
		error = EINVAL;
		try { SPDLOG_ERROR("JSON {} {} failed: {}", operation,
				   target ? target : "(null)", e.what()); } catch (...) {}
	} catch (...) {
		error = EIO;
		try { SPDLOG_ERROR("JSON {} {} failed", operation,
				   target ? target : "(null)"); } catch (...) {}
	}
	errno = error;
	return -1;
}
} // namespace

namespace bpftime
{
// The only code with access to the segment allocator is the transaction.
// Decoding/validation owns ordinary C++ values and cannot change shared state.
class shm_json_import {
	bpftime_shm &shm;
	std::optional<shm_write_window_guard> write_window;
	std::vector<std::pair<int, handler_variant>> staged;
	bool published = false;
	bool committed = false;

	bpf_perf_event_handler make_perf(const perf_definition &p)
	{
		const auto type = static_cast<bpf_event_type>(p.type);
		switch (type) {
		case bpf_event_type::BPF_TYPE_UPROBE:
		case bpf_event_type::BPF_TYPE_URETPROBE:
		case bpf_event_type::BPF_TYPE_UPROBE_OVERRIDE: {
			bpf_perf_event_handler h(type, p.offset, p.pid, p.name.c_str(),
						 shm.segment, p.enabled);
			std::get<uprobe_perf_event_data>(h.data).ref_ctr_off = p.ref_ctr_off;
			return h;
		}
		case bpf_event_type::PERF_TYPE_TRACEPOINT:
			return bpf_perf_event_handler(p.pid, p.tracepoint_id, shm.segment);
		case bpf_event_type::PERF_TYPE_SOFTWARE:
			return bpf_perf_event_handler(p.cpu, p.sample_type, p.config, shm.segment);
		default:
			invalid("unsupported perf event type");
		}
	}
public:
	explicit shm_json_import(bpftime_shm &shm) : shm(shm) {}
	~shm_json_import()
	{
		if (!committed && published)
			for (auto &[fd, handler] : staged)
				shm.manager->discard_handler(fd, shm.segment);
		for (auto &[fd, handler] : staged)
			if (auto *map = std::get_if<bpf_map_handler>(&handler))
				map->map_free(shm.segment);
	}
	void apply(const json &document)
	{
		if (!shm.manager) invalid("no handler manager in shared memory");
		const auto plan = decode_document(document, *shm.manager);
		staged.reserve(plan.size());
		write_window.emplace(shm);
#ifdef ENABLE_BPFTIME_VERIFIER
		std::optional<std::map<int, verifier::BpftimeMapDescriptor>> descriptors;
#endif
		for (const auto &[fd, value] : plan) {
			std::visit([&](const auto &d) {
				using T = std::decay_t<decltype(d)>;
				if constexpr (std::is_same_v<T, map_definition>) {
					staged.emplace_back(fd, bpf_map_handler(fd, d.name.c_str(), shm.segment, d.attr));
					// Register ownership before map_init: it can allocate an
					// implementation and then throw allocating its refcount.
					auto &map = std::get<bpf_map_handler>(staged.back().second);
					if (map.map_init(shm.segment) < 0)
						invalid("failed to initialize map at fd " + std::to_string(fd));
#ifdef ENABLE_BPFTIME_VERIFIER
					if (!descriptors)
						descriptors.emplace(verifier::get_map_descriptors());
					(*descriptors)[fd] = verifier::BpftimeMapDescriptor{
						.original_fd = fd,
						.type = static_cast<uint32_t>(d.attr.type),
						.key_size = d.attr.key_size,
						.value_size = d.attr.value_size,
						.max_entries = d.attr.max_ents,
						.inner_map_fd = static_cast<unsigned int>(-1)
					};
#endif
				} else if constexpr (std::is_same_v<T, program_definition>) {
					staged.emplace_back(fd, bpf_prog_handler(shm.segment, d.insns.data(),
										 d.insns.size(), d.name.c_str(), d.type));
				} else if constexpr (std::is_same_v<T, perf_definition>) {
					staged.emplace_back(fd, make_perf(d));
					std::get<bpf_perf_event_handler>(staged.back().second).enabled = d.enabled;
				} else if constexpr (std::is_same_v<T, link_definition>) {
					bpf_link_create_args args{};
					args.prog_fd = d.prog_fd;
					args.target_fd = d.target_fd;
					staged.emplace_back(fd, bpf_link_handler(args));
				}
			}, value);
		}
		if (shm.manager->install_prepared_handlers(staged, shm.segment) < 0)
			invalid("handler slots changed before import commit");
		published = true;
#ifdef ENABLE_BPFTIME_VERIFIER
		// This setter prepares a replacement and swaps only after all
		// allocation succeeds. If it throws, only our new handlers roll back.
		if (descriptors)
			verifier::set_map_descriptors(*descriptors);
#endif
		committed = true;
	}
};

int bpftime_import_shm_handler_from_json(bpftime_shm &shm, int fd, const char *text)
{
	return json_operation("load", "handler", [&] {
		if (!text || fd < 0) invalid("invalid handler fd or JSON pointer");
		const auto value = parse_document(text);
		shm_json_import transaction(shm);
		transaction.apply(json{ { std::to_string(fd), value } });
	});
}

int bpftime_import_shm_from_json(bpftime_shm &shm, const char *filename)
{
	return json_operation("import", filename, [&] {
		if (!filename) invalid("null JSON input filename");
		std::ifstream file(filename);
		if (!file.is_open()) io_error("open JSON import");
		const auto document = parse_document(file);
		if (file.bad()) io_error("read JSON import");
		shm_json_import transaction(shm);
		transaction.apply(document);
	});
}

} // namespace bpftime

extern "C" int bpftime_import_global_shm_from_json(const char *filename)
{
	if (!bpftime_global_shm_is_initialized()) {
		errno = ENODEV;
		return -1;
	}
	return bpftime::bpftime_import_shm_from_json(shm_holder.global_shared_memory, filename);
}

extern "C" int bpftime_import_shm_handler_from_json(int fd, const char *text)
{
	if (!bpftime_global_shm_is_initialized()) {
		errno = ENODEV;
		return -1;
	}
	return bpftime::bpftime_import_shm_handler_from_json(shm_holder.global_shared_memory, fd, text);
}

extern "C" int bpftime_export_global_shm_to_json(const char *filename)
{
	return bpftime_export_shm_to_json(shm_holder.global_shared_memory,
					  filename);
}

int bpftime::bpftime_export_shm_to_json(const bpftime_shm &shm,
					const char *filename)
{
	std::ofstream file(filename);
	json j;

	const handler_manager *manager = shm.get_manager();
	if (!manager) {
		SPDLOG_ERROR("No manager found in the shared memory");
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
			json attr = { { "type", prog_handler.type },
				      { "insns",
					buffer_to_hex_string(
						(const unsigned char *)insns,
						sizeof(ebpf_inst) * cnt) },
				      { "cnt", cnt } };
			j[std::to_string(i)] =
				json{ { "type", "bpf_prog_handler" },
				      { "attr", attr },
				      { "name", name } };
			// append attach fds to the json
			SPDLOG_INFO("find prog fd={} name={}", i,
				    prog_handler.name);
		} else if (std::holds_alternative<bpf_map_handler>(handler)) {
			auto &map_handler = std::get<bpf_map_handler>(handler);
			const char *name = map_handler.name.c_str();
			SPDLOG_INFO("bpf_map_handler name={} found at {}", name,
				    i);
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
				{ "attr", bpf_perf_event_handler_attr_to_json(
						  perf_handler) },
				{ "enabled", perf_handler.enabled }
			};
			SPDLOG_INFO("bpf_perf_event_handler found at {}", i);
		} else if (std::holds_alternative<epoll_handler>(handler)) {
			auto &h = std::get<epoll_handler>(handler);
			j[std::to_string(i)] = { { "type", "epoll_handler" } };
			SPDLOG_INFO("epoll_handler found at {}", i);
		} else if (std::holds_alternative<bpf_link_handler>(handler)) {
			auto &h = std::get<bpf_link_handler>(handler);
			j[std::to_string(i)] = {
				{ "type", "bpf_link_handler" },
				{ "attr",
				  { { "prog_fd", h.prog_id },
				    { "target_fd", h.attach_target_id } } }
			};
			SPDLOG_INFO(
				"bpf_link_handler found at {}，link {} -> {}",
				i, h.args.prog_fd, h.args.target_fd);
		} else if (std::holds_alternative<memfd_handler>(handler)) {
			auto &h = std::get<memfd_handler>(handler);
			j[std::to_string(i)] = { { "type", "memfd_handler" },
						 { "attr",
						   { { "flags", h.flags },
						     { "name", h.name } } } };

		} else {
			SPDLOG_ERROR("Unsupported handler type {}",
				     handler.index());
			return -1;
		}
	}
	// write the json to file
	file << j.dump(4);
	file.close();
	return 0;
}
