#include "rocm_attach_impl.hpp"
#include "rocm_attach_private_data.hpp"
#include "rocm_probe.hpp"
#include <atomic>
#include <climits>
#include <map>
#include <cerrno>
#include <spdlog/spdlog.h>

namespace bpftime::attach
{
namespace
{
struct registry {
	std::mutex mutex;
	std::map<int, std::shared_ptr<const rocm_probe>> probes;
	uint64_t generation = 0;
	int next_id = 1;
	std::atomic<bool> active{ false };
};
registry &probes()
{
	static auto *r = new registry;
	return *r;
}
} // namespace
bool has_rocm_probes() noexcept
{
	try {
		return probes().active.load();
	} catch (...) {
		return false;
	}
}
rocm_probe_snapshot snapshot_rocm_probes()
{
	auto &r = probes();
	std::lock_guard lock(r.mutex);
	rocm_probe_snapshot result{ r.generation, {} };
	for (auto &[id, probe] : r.probes)
		result.probes.push_back(probe);
	return result;
}
rocm_attach_impl::~rocm_attach_impl()
{
	while (!ids.empty())
		detach_by_id(*ids.begin());
}
int rocm_attach_impl::detach_by_id(int id)
{
	std::lock_guard lock(mutex);
	if (!ids.erase(id))
		return -ENOENT;
	auto &r = probes();
	std::lock_guard registry_lock(r.mutex);
	r.probes.erase(id);
	++r.generation;
	r.active.store(!r.probes.empty());
	return 0;
}

int rocm_attach_impl::create_attach_with_ebpf_callback(
	ebpf_run_callback &&, const attach_private_data &private_data,
	int attach_type)
{
	if (attach_type != ATTACH_ROCM_PROBE_AND_RETPROBE)
		return -ENOTSUP;
	auto *data =
		dynamic_cast<const rocm_attach_private_data *>(&private_data);
	if (!data || data->func_name.empty())
		return -EINVAL;
	try {
		std::lock_guard lock(mutex);
		auto &r = probes();
		std::lock_guard registry_lock(r.mutex);
		if (r.next_id == INT_MAX)
			return -EOVERFLOW;
		const int id = r.next_id++;
		auto probe = std::make_shared<rocm_probe>();
		probe->id = id;
		probe->kernel = data->func_name;
		probe->at_return = data->is_ret_probe;
		probe->entry = "bpftime_rocm_probe_" + std::to_string(id);
		probe->ir = compile_rocm_ebpf(data->instructions, probe->entry);
		ids.insert(id);
		try {
			r.probes.emplace(id, std::move(probe));
		} catch (...) {
			ids.erase(id);
			throw;
		}
		++r.generation;
		r.active.store(true);
		return id;
	} catch (const std::exception &error) {
		SPDLOG_ERROR("Cannot attach ROCm probe: {}", error.what());
		return -EINVAL;
	}
}
} // namespace bpftime::attach
