/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpftime_shm_json.hpp"
#include <json.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <atomic>
#include <charconv>
#include <climits>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <system_error>
#include <type_traits>
#include <fcntl.h>
#include <sys/stat.h>
#ifdef __linux__
#include <sys/syscall.h>
#endif
#include <unistd.h>

using namespace bpftime;
using json = nlohmann::json;

namespace
{
// JSON is a portable definition graph, not a live-memory checkpoint.
// Both import and export require callers to quiesce handler mutations.
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
	bool initialize;
};
struct program_definition {
	std::string name;
	int type;
	std::vector<ebpf_inst> insns;
	std::vector<uint8_t> aot;
	std::vector<int> legacy_targets;
};
struct perf_definition {
	int type;
	bool enabled;
	std::string data_type;
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
	uint32_t attach_type = 0;
	uint32_t flags = 0;
	std::optional<uint64_t> cookie;
};
using definition = std::variant<map_definition, program_definition,
				perf_definition, link_definition>;
using import_plan = std::map<int, definition>;

// Return whether staging needs a native implementation; callback-only maps
// retain metadata without acquiring or owning external resources.
bool validate_import_map(const bpf_map_attr &attr, bool allow_custom)
{
	// Only implementations whose resources are wholly owned by the shared
	// segment can be staged and rolled back. Lazy kernel map wrappers are
	// safe too, as are host-backed GPU arrays whose CUDA mapping is deferred.
	// Kernel perf transport, external lifecycle callbacks, and eager GPU creation
	// change state outside the segment and are not imported transactionally.
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
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_ARRAY:
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_HASH:
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_PERCPU_ARRAY:
	case bpf_map_type::BPF_MAP_TYPE_PROG_ARRAY:
	case bpf_map_type::BPF_MAP_TYPE_STACK_TRACE:
		break;
#endif
#ifdef BPFTIME_ENABLE_CUDA_ATTACH
	case bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP:
	case bpf_map_type::BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP:
		break;
#endif
	// Never publish a callback-only definition for a native dispatch type.
	// Unavailable built-ins remain reserved, rather than changing meaning
	// when the same document is imported by a differently configured build.
#ifndef BPFTIME_BUILD_WITH_LIBBPF
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_ARRAY:
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_HASH:
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_PERCPU_ARRAY:
	case bpf_map_type::BPF_MAP_TYPE_PROG_ARRAY:
	case bpf_map_type::BPF_MAP_TYPE_STACK_TRACE:
#endif
#ifndef BPFTIME_ENABLE_CUDA_ATTACH
	case bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP:
	case bpf_map_type::BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP:
#endif
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_PERF_EVENT_ARRAY:
	case bpf_map_type::BPF_MAP_TYPE_GPU_HASH_MAP:
	case bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_MAP:
	case bpf_map_type::BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP:
	case bpf_map_type::BPF_MAP_TYPE_GPU_KERNEL_SHARED_ARRAY_MAP:
	case bpf_map_type::BPF_MAP_TYPE_GPU_RINGBUF_MAP:
		invalid("map type cannot be restored transactionally by this build");
	default:
		if (allow_custom && attr.type >= 0 &&
		    attr.type < static_cast<int>(bpf_map_type::BPF_MAP_TYPE_MAX)) {
			const auto &ops = global_map_ops_table[attr.type];
			if (!ops.alloc_map && !ops.map_free &&
			    (ops.elem_lookup || ops.elem_update || ops.elem_delete ||
			     ops.map_get_next_key))
				return false; // Metadata only; never call external initialization.
		}
		invalid("custom map requires enabled callback-only operations");
	}
	const auto type = static_cast<bpf_map_type>(attr.type);
	if ((type == bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_ARRAY ||
	     type == bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_HASH ||
	     type == bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_PERCPU_ARRAY) &&
	    attr.kernel_bpf_map_id > static_cast<uint32_t>(INT_MAX))
		invalid("kernel map ID does not fit the implementation's signed ID");
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
		return true;
	}
	if (attr.value_size == 0)
		invalid("map value_size must be positive");
	const bool no_key = type == bpf_map_type::BPF_MAP_TYPE_QUEUE ||
			    type == bpf_map_type::BPF_MAP_TYPE_STACK ||
			    type == bpf_map_type::BPF_MAP_TYPE_BLOOM_FILTER;
	if ((no_key && attr.key_size != 0) || (!no_key && attr.key_size == 0))
		invalid("invalid map key_size");
	if ((type == bpf_map_type::BPF_MAP_TYPE_ARRAY ||
	     type == bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_ARRAY ||
	     type == bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_PERCPU_ARRAY ||
	     type == bpf_map_type::BPF_MAP_TYPE_PERCPU_ARRAY ||
	     type == bpf_map_type::BPF_MAP_TYPE_PERF_EVENT_ARRAY ||
	     type == bpf_map_type::BPF_MAP_TYPE_ARRAY_OF_MAPS ||
	     type == bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP ||
	     type == bpf_map_type::BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP ||
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
	case bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP:
		checked_product(attr.max_ents, attr.value_size,
				std::numeric_limits<size_t>::max());
		break;
	case bpf_map_type::BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP:
		// The constructor treats zero as one, but the public value-size
		// accessor uses the stored count. Reject that inconsistent definition.
		if (attr.gpu_thread_count == 0)
			invalid("host GPU array thread count must be positive");
		checked_product(attr.gpu_thread_count, attr.value_size, UINT32_MAX);
		checked_product(attr.max_ents, attr.gpu_thread_count * attr.value_size,
				std::numeric_limits<size_t>::max());
		break;
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_ARRAY:
		// The no-kernel fallback allocates with size_t but indexes with uint32_t.
		if (attr.kernel_bpf_map_id == 0)
			checked_product(attr.max_ents, attr.value_size, UINT32_MAX);
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
	return true;
}

json encode_map(const bpf_map_attr &a)
{
	// Definition export must not depend on this build's import capabilities.
	// In particular, preserve kernel/GPU/custom metadata without constructing
	// their external resources or promising transactional restoration.
	return { { "map_type", a.type }, { "key_size", a.key_size },
		 { "value_size", a.value_size }, { "max_entries", a.max_ents },
		 { "flags", a.flags }, { "ifindex", a.ifindex },
		 { "btf_vmlinux_value_type_id", a.btf_vmlinux_value_type_id },
		 { "btf_id", a.btf_id }, { "btf_key_type_id", a.btf_key_type_id },
		 { "btf_value_type_id", a.btf_value_type_id },
		 { "map_extra", a.map_extra },
		 { "kernel_bpf_map_id", a.kernel_bpf_map_id },
		 { "gpu_thread_count", a.gpu_thread_count } };
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
	a.gpu_thread_count = number<uint64_t>(j, "gpu_thread_count", 1024);
	return a;
}

void validate_link_type(uint32_t type)
{
	if (type != 0 && type != BPFTIME_BPF_PERF_EVENT_ATTACH_TYPE)
		invalid("unsupported link attach_type");
}

perf_definition decode_perf(const json &value)
{
	const auto &a = value.at("attr");
	perf_definition p{};
	p.type = integer<int>(a.at("type"));
	p.enabled = boolean(value.at("enabled"));
	const auto data_type = a.contains("data_type") ? string_field(a, "data_type") : "";
	// Custom event IDs are arbitrary integers and may overlap built-in IDs.
	// Their explicit payload discriminator takes precedence. Without one,
	// retain the legacy inference for built-in event types only.
	if (data_type == "custom_perf_event_data") {
		p.data_type = data_type;
		p.name = string_field(a, "attach_argument");
		return p;
	}
	switch (static_cast<bpf_event_type>(p.type)) {
	case bpf_event_type::BPF_TYPE_UPROBE:
	case bpf_event_type::BPF_TYPE_URETPROBE:
	case bpf_event_type::BPF_TYPE_UPROBE_OVERRIDE:
	case bpf_event_type::BPF_TYPE_UREPLACE:
		p.data_type = "uprobe_perf_event_data";
		p.pid = integer<int>(a.at("pid"));
		p.offset = integer<uint64_t>(a.at("offset"));
		p.ref_ctr_off = number<size_t>(a, "ref_ctr_off", 0);
		p.name = string_field(a, "_module_name");
		break;
	case bpf_event_type::BPF_TYPE_KPROBE:
	case bpf_event_type::BPF_TYPE_KRETPROBE:
		p.data_type = "kprobe_perf_event_data";
		p.offset = integer<uint64_t>(a.at("addr"));
		p.ref_ctr_off = number<size_t>(a, "ref_ctr_off", 0);
		p.name = string_field(a, "func_name");
		break;
	case bpf_event_type::PERF_TYPE_TRACEPOINT:
		p.data_type = "tracepoint_perf_event_data";
		p.pid = integer<int>(a.at("pid"));
		p.tracepoint_id = integer<int32_t>(a.at("tracepoint_id"));
		break;
	case bpf_event_type::PERF_TYPE_SOFTWARE:
		p.data_type = "software_perf_event_shared_ptr";
		p.cpu = integer<int>(a.at("cpu"));
		p.sample_type = integer<int32_t>(a.at("sample_type"));
		p.config = integer<int64_t>(a.at("config"));
		break;
	default:
		invalid("unknown perf event type requires custom_perf_event_data");
	}
	if (a.contains("data_type") && data_type != p.data_type)
		invalid("perf event type and data_type disagree");
	return p;
}

definition decode_handler(const json &value, bool allow_custom)
{
	const auto type = string_field(value, "type");
	if (type == "epoll_handler" || type == "memfd_handler")
		invalid("handler metadata is export-only: " + type);
	const auto &a = value.at("attr");
	if (!a.is_object())
		invalid("handler attr must be an object");
	if (type == "bpf_map_handler") {
		auto attr = decode_map(a);
		return map_definition{ string_field(value, "name"), attr,
				       validate_import_map(attr, allow_custom) };
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
		if (a.contains("aot_insns")) {
			const auto aot = string_field(a, "aot_insns");
			p.aot.resize(aot.size() / 2);
			if (hex_string_to_buffer(aot, p.aot.data(), p.aot.size()) < 0)
				invalid("invalid AOT hex");
		}
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
	if (type == "bpf_link_handler") {
		link_definition l{ integer<int>(a.at("prog_fd")),
				   integer<int>(a.at("target_fd")) };
		l.attach_type = number<uint32_t>(a, "attach_type", 0);
		l.flags = number<uint32_t>(a, "flags", 0);
		if (a.contains("attach_cookie") && !a.at("attach_cookie").is_null())
			l.cookie = integer<uint64_t>(a.at("attach_cookie"));
		validate_link_type(l.attach_type);
		return l;
	}
	invalid("unsupported handler type: " + type);
}

template <typename IsType>
void validate_link_endpoints(int fd, int program, int target, IsType &&is_type)
{
	if (!is_type(program, true) || !is_type(target, false))
		invalid("link " + std::to_string(fd) +
			" must reference a program and a perf event");
}

import_plan decode_document(const json &document, const handler_manager &manager,
			    bool allow_custom)
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
			plan.emplace(fd, decode_handler(value, allow_custom));
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
		if (fd < 0 || static_cast<size_t>(fd) >= manager.size())
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

json encode_perf(const bpf_perf_event_handler &handler)
{
	json a{ { "type", handler.type } };
	std::visit([&](const auto &data) {
		using T = std::decay_t<decltype(data)>;
		if constexpr (std::is_same_v<T, uprobe_perf_event_data>) {
			a["data_type"] = "uprobe_perf_event_data";
			a["offset"] = data.offset;
			a["pid"] = data.pid;
			a["ref_ctr_off"] = data.ref_ctr_off;
			a["_module_name"] = data._module_name;
		} else if constexpr (std::is_same_v<T, kprobe_perf_event_data>) {
			a["data_type"] = "kprobe_perf_event_data";
			a["func_name"] = data.func_name;
			a["addr"] = data.addr;
			a["ref_ctr_off"] = data.ref_ctr_off;
		} else if constexpr (std::is_same_v<T, tracepoint_perf_event_data>) {
			a["data_type"] = "tracepoint_perf_event_data";
			a["pid"] = data.pid;
			a["tracepoint_id"] = data.tracepoint_id;
		} else if constexpr (std::is_same_v<T, software_perf_event_shared_ptr>) {
			if (!data) invalid("null software perf event");
			a["data_type"] = "software_perf_event_shared_ptr";
			a["cpu"] = data->cpu;
			a["config"] = data->config;
			a["sample_type"] = data->sample_type;
		} else if constexpr (std::is_same_v<T, custom_perf_event_data>) {
			a["data_type"] = "custom_perf_event_data";
			a["attach_argument"] = data.attach_argument;
		}
	}, handler.data);
	return { { "type", "bpf_perf_event_handler" }, { "attr", a },
		 { "enabled", handler.enabled } };
}

json encode_handler(const handler_variant &handler)
{
	return std::visit([](const auto &h) -> json {
		using T = std::decay_t<decltype(h)>;
		if constexpr (std::is_same_v<T, bpf_map_handler>) {
			return { { "type", "bpf_map_handler" }, { "name", h.name },
				 { "attr", encode_map(h.attr) } };
		} else if constexpr (std::is_same_v<T, bpf_prog_handler>) {
			json a{ { "type", h.type }, { "cnt", h.insns.size() },
				{ "insns", buffer_to_hex_string(
					reinterpret_cast<const unsigned char *>(h.insns.data()),
					h.insns.size() * sizeof(ebpf_inst)) } };
			if (!h.aot_insns.empty())
				a["aot_insns"] = buffer_to_hex_string(h.aot_insns.data(),
								    h.aot_insns.size());
			return { { "type", "bpf_prog_handler" }, { "name", h.name }, { "attr", a } };
		} else if constexpr (std::is_same_v<T, bpf_perf_event_handler>) {
			return encode_perf(h);
		} else if constexpr (std::is_same_v<T, bpf_link_handler>) {
			validate_link_type(h.args.attach_type);
			return { { "type", "bpf_link_handler" },
				 { "attr", { { "prog_fd", h.prog_id },
					     { "target_fd", h.attach_target_id },
					     { "attach_type", h.args.attach_type },
					     { "flags", h.args.flags },
					     { "attach_cookie", h.attach_cookie ?
						     json(*h.attach_cookie) : json(nullptr) } } } };
		} else if constexpr (std::is_same_v<T, epoll_handler>) {
			// Preserve the legacy inspection record, not its registrations.
			return { { "type", "epoll_handler" } };
		} else if constexpr (std::is_same_v<T, memfd_handler>) {
			// Name and flags are metadata only; no backing contents are saved.
			return { { "type", "memfd_handler" },
				 { "attr", { { "flags", h.flags }, { "name", h.name } } } };
		} else {
			invalid("unsupported handler type for JSON export");
		}
	}, handler);
}

// These descriptors belong to JSON file I/O, never to the handler registry.
// The Linux syscall server interposes close(), but forwards SYS_close directly
// to libc. Avoid clearing a shared handler whose ID happens to match an OS fd.
int close_os_fd(int value)
{
#ifdef __linux__
	return static_cast<int>(::syscall(SYS_close, static_cast<long>(value),
					 0L, 0L, 0L, 0L, 0L));
#else
	return ::close(value);
#endif
}

class owned_fd {
	int value_;
public:
	explicit owned_fd(int value) : value_(value) {}
	~owned_fd()
	{
		const int saved_errno = errno;
		if (value_ >= 0) close_os_fd(value_);
		errno = saved_errno;
	}
	owned_fd(const owned_fd &) = delete;
	owned_fd &operator=(const owned_fd &) = delete;
	int get() const { return value_; }
	void close()
	{
		int value = value_;
		value_ = -1;
		if (close_os_fd(value) < 0)
			throw std::system_error(errno, std::generic_category(), "close JSON export");
	}
};

[[noreturn]] void io_error(const char *operation)
{
	throw std::system_error(errno, std::generic_category(), operation);
}

struct destination_state {
	bool exists = false;
	struct stat status{};
};

destination_state inspect_destination(int directory, const std::string &name)
{
	destination_state state;
	if (fstatat(directory, name.c_str(), &state.status, AT_SYMLINK_NOFOLLOW) == 0) {
		state.exists = true;
		if (!S_ISREG(state.status.st_mode) || state.status.st_nlink != 1)
			invalid("JSON export requires a regular file with no symlinks or hard links");
	} else if (errno != ENOENT) {
		io_error("inspect JSON export destination");
	}
	return state;
}

void atomic_write(const bpftime_shm &shm, const char *filename, const std::string &text)
{
	std::filesystem::path path(filename);
	auto parent = path.parent_path();
	if (parent.empty()) parent = ".";
	const auto name = path.filename().string();
	if (name.empty() || name == "." || name == "..")
		invalid("invalid JSON export filename");
	owned_fd directory(open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
	if (directory.get() < 0) io_error("open JSON export directory");

#ifdef __linux__
	// Check the directory identity, not path spelling, so ./ and symlinked
	// parents cannot replace the active segment's name.
	struct stat parent_stat{}, shm_directory_stat{};
	if (fstat(directory.get(), &parent_stat) < 0)
		io_error("inspect JSON export directory");
	std::string segment_name = shm.get_segment_name();
	while (!segment_name.empty() && segment_name.front() == '/')
		segment_name.erase(segment_name.begin());
	if (stat("/dev/shm", &shm_directory_stat) == 0 &&
	    parent_stat.st_dev == shm_directory_stat.st_dev &&
	    parent_stat.st_ino == shm_directory_stat.st_ino && name == segment_name)
		invalid("JSON export destination is the active shared memory segment");
#endif
	const auto before = inspect_destination(directory.get(), name);
	static std::atomic<uint64_t> sequence{ 0 };
	std::string temporary;
	int fd = -1;
	for (int attempts = 0; attempts != 64; ++attempts) {
		temporary = ".bpftime-json-" + std::to_string(getpid()) + "-" +
			    std::to_string(sequence.fetch_add(1));
		if (temporary == name) continue;
		fd = openat(directory.get(), temporary.c_str(),
			    O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
		if (fd >= 0) break;
		if (errno != EEXIST) io_error("open JSON export temporary file");
	}
	if (fd < 0) io_error("create unique JSON export temporary file");
	owned_fd file(fd);
	struct temporary_cleanup {
		int directory;
		const std::string &name;
		bool active = true;
		~temporary_cleanup() { if (active) unlinkat(directory, name.c_str(), 0); }
	} cleanup{ directory.get(), temporary };
	if (before.exists && fchmod(fd, before.status.st_mode & 0777) < 0)
		io_error("set JSON export permissions");
	for (size_t offset = 0; offset != text.size();) {
		ssize_t written = write(fd, text.data() + offset,
					std::min(text.size() - offset,
						 static_cast<size_t>(SSIZE_MAX)));
		if (written < 0 && errno == EINTR) continue;
		if (written <= 0) {
			if (written == 0) errno = EIO;
			io_error("write JSON export");
		}
		offset += static_cast<size_t>(written);
	}
	if (fsync(fd) < 0) io_error("flush JSON export");
	file.close();
	const auto after = inspect_destination(directory.get(), name);
	if (before.exists != after.exists ||
	    (before.exists && (before.status.st_dev != after.status.st_dev ||
			       before.status.st_ino != after.status.st_ino)))
		invalid("JSON export destination changed while writing");
	if (renameat(directory.get(), temporary.c_str(), directory.get(), name.c_str()) < 0)
		io_error("replace JSON export");
	cleanup.active = false;
	// Rename is the commit point. There are no fallible operations after it.
	// Atomic visibility is guaranteed, not crash durability of the directory.
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
	std::vector<std::pair<int, handler_variant>> staged;
	bool published = false;
	bool committed = false;

	bpf_perf_event_handler make_perf(const perf_definition &p)
	{
		const auto type = static_cast<bpf_event_type>(p.type);
		if (p.data_type == "uprobe_perf_event_data") {
			bpf_perf_event_handler h(type, p.offset, p.pid, p.name.c_str(),
						 shm.segment, p.enabled);
			std::get<uprobe_perf_event_data>(h.data).ref_ctr_off = p.ref_ctr_off;
			return h;
		}
		if (p.data_type == "kprobe_perf_event_data")
			return bpf_perf_event_handler(type == bpf_event_type::BPF_TYPE_KRETPROBE,
						      p.offset, p.name.c_str(), p.ref_ctr_off, shm.segment);
		if (p.data_type == "tracepoint_perf_event_data")
			return bpf_perf_event_handler(p.pid, p.tracepoint_id, shm.segment);
		if (p.data_type == "software_perf_event_shared_ptr")
			return bpf_perf_event_handler(p.cpu, p.sample_type, p.config, shm.segment);
		return bpf_perf_event_handler(p.type, p.name.c_str(), shm.segment);
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
		const auto plan = decode_document(document, *shm.manager,
			shm.get_runtime_config().allow_non_buildin_map_types);
		staged.reserve(plan.size());
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
					// Callback-only maps retain a null implementation. map_free
					// returns immediately for them, even if callbacks later change.
					if (d.initialize && map.map_init(shm.segment) < 0)
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
					auto &p = std::get<bpf_prog_handler>(staged.back().second);
					p.aot_insns.assign(d.aot.begin(), d.aot.end());
				} else if constexpr (std::is_same_v<T, perf_definition>) {
					staged.emplace_back(fd, make_perf(d));
					std::get<bpf_perf_event_handler>(staged.back().second).enabled = d.enabled;
				} else if constexpr (std::is_same_v<T, link_definition>) {
					bpf_link_create_args args{};
					args.prog_fd = d.prog_fd;
					args.target_fd = d.target_fd;
					args.attach_type = d.attach_type;
					args.flags = d.flags;
					if (d.attach_type == BPFTIME_BPF_PERF_EVENT_ATTACH_TYPE)
						args.perf_event.bpf_cookie = d.cookie.value_or(0);
					staged.emplace_back(fd, bpf_link_handler(args));
					std::get<bpf_link_handler>(staged.back().second).attach_cookie = d.cookie;
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

int bpftime_export_shm_to_json(const bpftime_shm &shm, const char *filename)
{
	return json_operation("export", filename, [&] {
		if (!filename) invalid("null JSON export filename");
		const auto *manager = shm.get_manager();
		if (!manager) invalid("no handler manager in shared memory");
		json document = json::object();
		auto is_type = [&](int fd, bool program) {
			const auto &handler = manager->get_handler(fd);
			return program ? std::holds_alternative<bpf_prog_handler>(handler) :
					 std::holds_alternative<bpf_perf_event_handler>(handler);
		};
		for (size_t fd = 0; fd != manager->size(); ++fd) {
			if (!manager->is_allocated(static_cast<int>(fd)))
				continue;
			const auto &handler = manager->get_handler(static_cast<int>(fd));
			if (const auto *link = std::get_if<bpf_link_handler>(&handler))
				validate_link_endpoints(static_cast<int>(fd), link->prog_id,
						link->attach_target_id, is_type);
			document[std::to_string(fd)] = encode_handler(handler);
		}
		// Validate graph references directly. Decoding our output would copy
		// all bytecode and incorrectly apply import-only capability checks.
		const auto text = document.dump(4);
		atomic_write(shm, filename, text);
	});
}
} // namespace bpftime

extern "C" int bpftime_import_global_shm_from_json(const char *filename)
{
	return bpftime::bpftime_import_shm_from_json(shm_holder.global_shared_memory, filename);
}

extern "C" int bpftime_import_shm_handler_from_json(int fd, const char *text)
{
	return bpftime::bpftime_import_shm_handler_from_json(shm_holder.global_shared_memory, fd, text);
}

extern "C" int bpftime_export_global_shm_to_json(const char *filename)
{
	return bpftime::bpftime_export_shm_to_json(shm_holder.global_shared_memory, filename);
}
