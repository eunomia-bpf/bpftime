#include "nv_gpu_gdrcopy.hpp"

#include "nv_gpu_gdrcopy_abi.hpp"

#include "bpftime_shm.hpp"
#include "spdlog/spdlog.h"

#include <cstddef>

#if defined(BPFTIME_ENABLE_GDRCOPY)
#include <array>
#include <dlfcn.h>
#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace bpftime
{
namespace gpu
{
namespace gdrcopy
{
namespace
{
using gdr_t = detail::gdr_t;
using gdr_mh_t = detail::gdr_mh_t;
using gdr_info_t = detail::gdr_info_t;
constexpr uint32_t kGdrPinDefaultFlags = detail::kGdrPinDefaultFlags;

constexpr std::array<const char *, 2> kGdrcopyLibraryNames = {
	"libgdrapi.so",
	"libgdrapi.so.2",
};

// Function table for libgdrapi loaded via dlopen so that
// binaries can still run on systems without GDRCopy installed.
struct gdrcopy_function_table {
	void *lib_handle = nullptr;
	bool initialized = false;

	gdr_t (*gdr_open_fn)() = nullptr;
	int (*gdr_close_fn)(gdr_t) = nullptr;
	int (*gdr_pin_buffer_fn)(gdr_t, unsigned long, size_t, uint64_t,
				 uint32_t, gdr_mh_t *) = nullptr;
	int (*gdr_pin_buffer_v2_fn)(gdr_t, unsigned long, size_t, uint32_t,
				    gdr_mh_t *) = nullptr;
	int (*gdr_unpin_buffer_fn)(gdr_t, gdr_mh_t) = nullptr;
	int (*gdr_map_fn)(gdr_t, gdr_mh_t, void **, size_t) = nullptr;
	int (*gdr_unmap_fn)(gdr_t, gdr_mh_t, void *, size_t) = nullptr;
	int (*gdr_get_info_fn)(gdr_t, gdr_mh_t, gdr_info_t *) = nullptr;
	int (*gdr_copy_from_mapping_fn)(gdr_mh_t, void *, const void *,
					size_t) = nullptr;
};

gdrcopy_function_table &get_gdrcopy_table()
{
	static gdrcopy_function_table function_table;
	return function_table;
}

bool ensure_gdrcopy_symbols_loaded(gdrcopy_function_table &function_table,
				   const char *map_type_name)
{
	if (function_table.initialized) {
		return function_table.lib_handle != nullptr;
	}

	for (const char *library_name : kGdrcopyLibraryNames) {
		function_table.lib_handle =
			dlopen(library_name, RTLD_LAZY | RTLD_LOCAL);
		if (function_table.lib_handle) {
			break;
		}
	}

	if (!function_table.lib_handle) {
		SPDLOG_INFO(
			"GDRCopy requested for {} but libgdrapi.so is not available; falling back to cuMemcpy",
			map_type_name);
		function_table.initialized = true;
		return false;
	}

	auto load = [&](auto &fn, const char *symbol) -> bool {
		fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(
			dlsym(function_table.lib_handle, symbol));
		if (!fn) {
			SPDLOG_ERROR(
				"Failed to resolve {} from libgdrapi.so",
				symbol);
			return false;
		}
		return true;
	};

	bool ok = true;
	ok &= load(function_table.gdr_open_fn, "gdr_open");
	ok &= load(function_table.gdr_close_fn, "gdr_close");
	ok &= load(function_table.gdr_pin_buffer_fn, "gdr_pin_buffer");
	// Optional helpers
	function_table.gdr_pin_buffer_v2_fn =
		reinterpret_cast<decltype(function_table.gdr_pin_buffer_v2_fn)>(
			dlsym(function_table.lib_handle, "gdr_pin_buffer_v2"));
	ok &= load(function_table.gdr_unpin_buffer_fn, "gdr_unpin_buffer");
	ok &= load(function_table.gdr_map_fn, "gdr_map");
	ok &= load(function_table.gdr_unmap_fn, "gdr_unmap");
	ok &= load(function_table.gdr_get_info_fn, "gdr_get_info");
	ok &= load(function_table.gdr_copy_from_mapping_fn,
		  "gdr_copy_from_mapping");

	if (!ok) {
		SPDLOG_ERROR(
			"libgdrapi.so found but some required symbols are missing; disabling GDRCopy fast-path");
		dlclose(function_table.lib_handle);
		function_table.lib_handle = nullptr;
		function_table.initialized = true;
		return false;
	}

	function_table.initialized = true;
	return true;
}

struct gdrcopy_mapping_state {
	bool attempted = false;
	bool enabled = false;

	gdr_t gdr_context = nullptr;
	gdr_mh_t gdr_memory_handle{};
	void *mapped_bar_address = nullptr;
	uint8_t *mapped_cpu_base = nullptr;
};

std::unordered_map<const void *, gdrcopy_mapping_state> &
get_mapping_state_by_mapping_key()
{
	static std::unordered_map<const void *, gdrcopy_mapping_state>
		state_by_mapping_key;
	return state_by_mapping_key;
}

std::mutex &get_mapping_state_mutex()
{
	static std::mutex mutex;
	return mutex;
}

bool gdrcopy_enabled_for_per_key_bytes(uint64_t per_key_bytes)
{
	const auto &config = bpftime::bpftime_get_agent_config();
	if (!config.enable_gpu_gdrcopy) {
		return false;
	}
	if (config.gpu_gdrcopy_max_per_key_bytes != 0 &&
	    per_key_bytes > config.gpu_gdrcopy_max_per_key_bytes) {
		return false;
	}
	return true;
}

} // namespace

bool copy_from_device_to_host_with_gdrcopy(const void *mapping_key,
					   const char *map_type_name,
					   CUdeviceptr device_base,
					   uint64_t total_buffer_size,
					   uint64_t copy_offset_bytes,
					   void *destination,
					   uint64_t copy_size_bytes,
					   uint64_t per_key_bytes)
{
	if (!mapping_key || !map_type_name || !destination) {
		return false;
	}
	if (copy_offset_bytes + copy_size_bytes > total_buffer_size) {
		return false;
	}

	std::lock_guard<std::mutex> guard(get_mapping_state_mutex());
	auto &state = get_mapping_state_by_mapping_key()[mapping_key];

	if (!state.attempted) {
		state.attempted = true;

		if (!gdrcopy_enabled_for_per_key_bytes(per_key_bytes)) {
			return false;
		}

		auto &function_table = get_gdrcopy_table();
		if (!ensure_gdrcopy_symbols_loaded(function_table,
						   map_type_name)) {
			return false;
		}

		auto gdr_context_local = function_table.gdr_open_fn();
		if (!gdr_context_local) {
			SPDLOG_ERROR(
				"gdr_open() failed for {}; falling back to cuMemcpy",
				map_type_name);
			return false;
		}

		gdr_mh_t gdr_memory_handle_local{};
		int gdr_status = -1;
		if (function_table.gdr_pin_buffer_v2_fn) {
			gdr_status = function_table.gdr_pin_buffer_v2_fn(
				gdr_context_local,
				static_cast<unsigned long>(device_base),
				total_buffer_size, kGdrPinDefaultFlags,
				&gdr_memory_handle_local);
		}
		if (gdr_status != 0) {
			gdr_status = function_table.gdr_pin_buffer_fn(
				gdr_context_local,
				static_cast<unsigned long>(device_base),
				total_buffer_size, 0, 0,
				&gdr_memory_handle_local);
		}

		if (gdr_status != 0) {
			SPDLOG_WARN(
				"gdr_pin_buffer() failed for {} (status={}); disabling GDRCopy path",
				map_type_name, gdr_status);
			function_table.gdr_close_fn(gdr_context_local);
			return false;
		}

		void *mapped_bar_address_local = nullptr;
		gdr_status = function_table.gdr_map_fn(
			gdr_context_local, gdr_memory_handle_local,
			&mapped_bar_address_local, total_buffer_size);
		if (gdr_status != 0) {
			SPDLOG_WARN(
				"gdr_map() failed for {} (status={}); disabling GDRCopy path",
				map_type_name, gdr_status);
			function_table.gdr_unpin_buffer_fn(
				gdr_context_local, gdr_memory_handle_local);
			function_table.gdr_close_fn(gdr_context_local);
			return false;
		}

		gdr_info_t mapping_info{};
		gdr_status = function_table.gdr_get_info_fn(
			gdr_context_local, gdr_memory_handle_local,
			&mapping_info);
		if (gdr_status != 0) {
			SPDLOG_WARN(
				"gdr_get_info() failed for {} (status={}); disabling GDRCopy path",
				map_type_name, gdr_status);
			function_table.gdr_unmap_fn(
				gdr_context_local, gdr_memory_handle_local,
				mapped_bar_address_local, total_buffer_size);
			function_table.gdr_unpin_buffer_fn(
				gdr_context_local, gdr_memory_handle_local);
			function_table.gdr_close_fn(gdr_context_local);
			return false;
		}

		auto base_offset = static_cast<uintptr_t>(device_base) -
				   static_cast<uintptr_t>(mapping_info.va);
		if (base_offset >= mapping_info.mapped_size) {
			SPDLOG_WARN(
				"gdrcopy mapping for {} returned unexpected VA range (base={:x}, va={:x}, mapped_size={}); disabling GDRCopy path",
				map_type_name, (uintptr_t)device_base,
				(uintptr_t)mapping_info.va,
				(uint64_t)mapping_info.mapped_size);
			function_table.gdr_unmap_fn(
				gdr_context_local, gdr_memory_handle_local,
				mapped_bar_address_local, total_buffer_size);
			function_table.gdr_unpin_buffer_fn(
				gdr_context_local, gdr_memory_handle_local);
			function_table.gdr_close_fn(gdr_context_local);
			return false;
		}

		auto *mapped_cpu_base_local =
			static_cast<uint8_t *>(mapped_bar_address_local) +
			base_offset;

		state.gdr_context = gdr_context_local;
		state.gdr_memory_handle = gdr_memory_handle_local;
		state.mapped_bar_address = mapped_bar_address_local;
		state.mapped_cpu_base = mapped_cpu_base_local;
		state.enabled = true;

		SPDLOG_INFO(
			"gdrcopy enabled for {}: device_base={:x}, mapped_cpu_base={:x}, total_size={}, mapped_size={}",
			map_type_name, (uintptr_t)device_base,
			(uintptr_t)mapped_cpu_base_local, total_buffer_size,
			(uint64_t)mapping_info.mapped_size);
	}

	if (!state.enabled) {
		return false;
	}

	auto &function_table = get_gdrcopy_table();
	auto *source = state.mapped_cpu_base + copy_offset_bytes;
	int gdr_status = function_table.gdr_copy_from_mapping_fn(
		state.gdr_memory_handle, destination, source, copy_size_bytes);
	if (gdr_status != 0) {
		SPDLOG_WARN(
			"gdr_copy_from_mapping failed for {} (status={}); falling back to cuMemcpy",
			map_type_name, gdr_status);
		return false;
	}

	return true;
}

void destroy_gdrcopy_mapping_for_owner(const void *mapping_key,
				       const char *map_type_name,
				       uint64_t total_buffer_size)
{
	if (!mapping_key || !map_type_name) {
		return;
	}

	std::lock_guard<std::mutex> guard(get_mapping_state_mutex());
	auto &state_by_mapping_key = get_mapping_state_by_mapping_key();

	auto it = state_by_mapping_key.find(mapping_key);
	if (it == state_by_mapping_key.end()) {
		return;
	}
	auto &state = it->second;
	if (!state.enabled || !state.gdr_context) {
		state_by_mapping_key.erase(it);
		return;
	}

	auto &function_table = get_gdrcopy_table();
	if (function_table.gdr_unmap_fn) {
		function_table.gdr_unmap_fn(state.gdr_context,
					    state.gdr_memory_handle,
					    state.mapped_bar_address,
					    total_buffer_size);
	}
	if (function_table.gdr_unpin_buffer_fn) {
		function_table.gdr_unpin_buffer_fn(state.gdr_context,
						   state.gdr_memory_handle);
	}
	if (function_table.gdr_close_fn) {
		function_table.gdr_close_fn(state.gdr_context);
	}

	state_by_mapping_key.erase(it);
}

} // namespace gdrcopy
} // namespace gpu
} // namespace bpftime

#else

namespace bpftime
{
namespace gpu
{
namespace gdrcopy
{
bool copy_from_device_to_host_with_gdrcopy(const void *, const char *,
					   CUdeviceptr, uint64_t, uint64_t,
					   void *, uint64_t, uint64_t)
{
	return false;
}

void destroy_gdrcopy_mapping_for_owner(const void *, const char *, uint64_t)
{
}
} // namespace gdrcopy
} // namespace gpu
} // namespace bpftime

#endif
