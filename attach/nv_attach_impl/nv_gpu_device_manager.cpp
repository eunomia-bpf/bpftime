#include "nv_gpu_device_manager.hpp"
#include <cstdlib>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace bpftime
{
namespace attach
{

void gpu_device_manager::initialize()
{
	devices_.clear();
	CUresult err = cuInit(0);
	if (err != CUDA_SUCCESS) {
		SPDLOG_WARN(
			"Failed to initialize CUDA driver ({}), assuming 0 devices",
			(int)err);
		return;
	}

	int count = 0;
	err = cuDeviceGetCount(&count);
	if (err != CUDA_SUCCESS) {
		SPDLOG_WARN("Failed to get CUDA device count ({})", (int)err);
		return;
	}

	SPDLOG_INFO("Detected {} CUDA device(s)", count);

	// Check if user overrides SM arch for all devices
	const char *sm_arch_override = std::getenv("BPFTIME_SM_ARCH");

	devices_.reserve(count);
	for (int i = 0; i < count; i++) {
		gpu_device_info info;
		info.device_ordinal = i;

		err = cuDeviceGet(&info.cu_device, i);
		if (err != CUDA_SUCCESS) {
			SPDLOG_ERROR("Failed to get CUDA device {} ({})", i,
				     (int)err);
			continue;
		}

		if (sm_arch_override && sm_arch_override[0] != '\0') {
			info.sm_arch = sm_arch_override;
		} else {
			int major = 0, minor = 0;
			cuDeviceGetAttribute(
				&major,
				CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
				info.cu_device);
			cuDeviceGetAttribute(
				&minor,
				CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
				info.cu_device);
			info.sm_arch =
				"sm_" + std::to_string(major * 10 + minor);
		}

		info.module_pool = std::make_shared<
			std::map<std::string, std::shared_ptr<ptx_in_module>>>();

		SPDLOG_INFO("GPU device {}: {} (ordinal {})", i, info.sm_arch,
			    info.device_ordinal);
		devices_.push_back(std::move(info));
	}
}

gpu_device_info &gpu_device_manager::get_device(int ordinal)
{
	if (ordinal < 0 || ordinal >= (int)devices_.size()) {
		SPDLOG_ERROR("Invalid device ordinal: {} (have {} devices)",
			     ordinal, devices_.size());
		throw std::out_of_range("Invalid GPU device ordinal");
	}
	return devices_[ordinal];
}

const gpu_device_info &gpu_device_manager::get_device(int ordinal) const
{
	if (ordinal < 0 || ordinal >= (int)devices_.size()) {
		throw std::out_of_range("Invalid GPU device ordinal");
	}
	return devices_[ordinal];
}

int gpu_device_manager::device_count() const
{
	return (int)devices_.size();
}

const std::vector<gpu_device_info> &gpu_device_manager::devices() const
{
	return devices_;
}

} // namespace attach
} // namespace bpftime
