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
	CUresult err = cuInit(0);
	if (err != CUDA_SUCCESS) {
		SPDLOG_WARN(
			"Failed to initialize CUDA driver ({}), assuming 0 devices",
			(int)err);
		count_ = 0;
		return;
	}

	err = cuDeviceGetCount(&count_);
	if (err != CUDA_SUCCESS) {
		SPDLOG_WARN("Failed to get CUDA device count ({})", (int)err);
		count_ = 0;
		return;
	}

	SPDLOG_INFO("Detected {} CUDA device(s)", count_);

	// Check if user overrides SM arch for all devices
	const char *sm_arch_override = std::getenv("BPFTIME_SM_ARCH");

	devices_.reserve(count_);
	for (int i = 0; i < count_; i++) {
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

gpu_device_info &gpu_device_manager::get_current_device()
{
	CUdevice device;
	CUresult err = cuCtxGetDevice(&device);
	if (err == CUDA_SUCCESS) {
		for (auto &info : devices_) {
			if (info.cu_device == device) {
				return info;
			}
		}
	}
	// Fall back to device 0
	if (!devices_.empty()) {
		return devices_[0];
	}
	throw std::runtime_error(
		"No CUDA devices available in gpu_device_manager");
}

std::set<std::string> gpu_device_manager::get_unique_sm_archs() const
{
	std::set<std::string> archs;
	for (const auto &info : devices_) {
		archs.insert(info.sm_arch);
	}
	return archs;
}

int gpu_device_manager::device_count() const
{
	return (int)devices_.size();
}

gpu_device_info &gpu_device_manager::get_default_device()
{
	return get_device(0);
}

std::vector<gpu_device_info> &gpu_device_manager::devices()
{
	return devices_;
}

const std::vector<gpu_device_info> &gpu_device_manager::devices() const
{
	return devices_;
}

} // namespace attach
} // namespace bpftime
