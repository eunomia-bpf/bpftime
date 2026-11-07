#include "bpftime_shm_internal.hpp"
#include <bpf_attach_ctx.hpp>
#include <spdlog/spdlog.h>

namespace bpftime
{
std::vector<attach::MapBasicInfo>
bpf_attach_ctx::create_map_basic_info(int filled_size)
{
	std::vector<attach::MapBasicInfo> local_basic_info(filled_size);
	for (auto &entry : local_basic_info) {
		entry.enabled = false;
		entry.key_size = 0;
		entry.value_size = 0;
		entry.max_entries = 0;
		entry.map_type = 0;
		entry.extra_buffer = nullptr;
		entry.max_thread_count = 0;
	}
	const auto &handler_manager =
		*shm_holder.global_shared_memory.get_manager();
	for (size_t i = 0; i < handler_manager.size(); i++) {
		const auto &current_handler = handler_manager.get_handler(i);
		if (std::holds_alternative<bpf_map_handler>(current_handler)) {
			if (i >= local_basic_info.size()) {
				SPDLOG_ERROR(
					"Too large map fd: {}, max to be {}", i,
					local_basic_info.size());
				return {};
			}
			auto &local = local_basic_info[i];
			const auto &map =
				std::get<bpf_map_handler>(current_handler);
			void *gpu_buffer = nullptr;
#ifdef BPFTIME_ENABLE_CUDA_ATTACH
			gpu_buffer = map.get_gpu_map_extra_buffer();
#endif
			SPDLOG_INFO(
				"Copying map fd {} to device, key size={}, value size={}, max ent={}, map_type = {}, gpu_buffer = {:x}",
				i, map.get_key_size(), map.get_value_size(),
				map.get_max_entries(), (int)map.type,
				(uintptr_t)gpu_buffer);

			local.enabled = true;
			local.key_size = map.get_key_size();
			local.value_size = map.get_value_size();
			local.max_entries = map.get_max_entries();
			local.map_type = (int)map.type;
			local.extra_buffer = gpu_buffer;
#ifdef BPFTIME_ENABLE_CUDA_ATTACH
			local.max_thread_count =
				map.get_gpu_map_max_thread_count();
#endif
		}
	}

	return local_basic_info;
}

} // namespace bpftime
