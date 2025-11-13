#include "nv_attach_fatbin_record.hpp"
#include "cuda.h"
#include "spdlog/spdlog.h"
#include "nv_attach_impl.hpp"
#include <array>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string_view>
#include <algorithm>
#include <string>
#define CUDA_DRIVER_CHECK_NO_EXCEPTION(expr, message)                          \
	do {                                                                   \
		if (auto err = expr; err != CUDA_SUCCESS) {                    \
			SPDLOG_ERROR("{}: {}", message, (int)err);             \
		}                                                              \
	} while (false)
#define CUDA_DRIVER_CHECK_EXCEPTION(expr, message)                             \
	do {                                                                   \
		if (auto err = expr; err != CUDA_SUCCESS) {                    \
			SPDLOG_ERROR("{}: {}", message, (int)err);             \
			throw std::runtime_error(message);                     \
		}                                                              \
	} while (false)

namespace bpftime::attach
{
namespace
{
std::optional<CUjit_target> to_jit_target(int value, bool accelerated)
{
	if (accelerated) {
		if (value == 90)
			return CU_TARGET_COMPUTE_90A;
		return std::nullopt;
	}
	switch (value) {
	case 30:
		return CU_TARGET_COMPUTE_30;
	case 32:
		return CU_TARGET_COMPUTE_32;
	case 35:
		return CU_TARGET_COMPUTE_35;
	case 37:
		return CU_TARGET_COMPUTE_37;
	case 50:
		return CU_TARGET_COMPUTE_50;
	case 52:
		return CU_TARGET_COMPUTE_52;
	case 53:
		return CU_TARGET_COMPUTE_53;
	case 60:
		return CU_TARGET_COMPUTE_60;
	case 61:
		return CU_TARGET_COMPUTE_61;
	case 62:
		return CU_TARGET_COMPUTE_62;
	case 70:
		return CU_TARGET_COMPUTE_70;
	case 72:
		return CU_TARGET_COMPUTE_72;
	case 75:
		return CU_TARGET_COMPUTE_75;
	case 80:
		return CU_TARGET_COMPUTE_80;
	case 86:
		return CU_TARGET_COMPUTE_86;
	case 87:
		return CU_TARGET_COMPUTE_87;
	case 89:
		return CU_TARGET_COMPUTE_89;
	case 90:
		return CU_TARGET_COMPUTE_90;
	default:
		return std::nullopt;
	}
}
std::optional<CUjit_target> device_default_target()
{
    if (auto err = cuInit(0); err != CUDA_SUCCESS) {
        SPDLOG_DEBUG("cuInit failed while probing device target: {}", (int)err);
        return std::nullopt;
    }
	CUdevice dev;
	if (auto err = cuDeviceGet(&dev, 0); err != CUDA_SUCCESS) {
		SPDLOG_DEBUG("cuDeviceGet failed while probing device target: {}", (int)err);
		return std::nullopt;
	}
	int major = 0, minor = 0;
	if (cuDeviceGetAttribute(&major,
	     CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) != CUDA_SUCCESS ||
	    cuDeviceGetAttribute(&minor,
	     CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) != CUDA_SUCCESS) {
		SPDLOG_DEBUG("cuDeviceGetAttribute failed while probing device target");
		return std::nullopt;
	}
	return to_jit_target(major * 10 + minor, false);
}

static std::optional<int> sm_from_jit_target(CUjit_target t)
{
    switch (t) {
    case CU_TARGET_COMPUTE_30:
        return 30;
    case CU_TARGET_COMPUTE_32:
        return 32;
    case CU_TARGET_COMPUTE_35:
        return 35;
    case CU_TARGET_COMPUTE_37:
        return 37;
    case CU_TARGET_COMPUTE_50:
        return 50;
    case CU_TARGET_COMPUTE_52:
        return 52;
    case CU_TARGET_COMPUTE_53:
        return 53;
    case CU_TARGET_COMPUTE_60:
        return 60;
    case CU_TARGET_COMPUTE_61:
        return 61;
    case CU_TARGET_COMPUTE_62:
        return 62;
    case CU_TARGET_COMPUTE_70:
        return 70;
    case CU_TARGET_COMPUTE_72:
        return 72;
    case CU_TARGET_COMPUTE_75:
        return 75;
    case CU_TARGET_COMPUTE_80:
        return 80;
    case CU_TARGET_COMPUTE_86:
        return 86;
    case CU_TARGET_COMPUTE_87:
        return 87;
    case CU_TARGET_COMPUTE_89:
        return 89;
    case CU_TARGET_COMPUTE_90:
        return 90;
    case CU_TARGET_COMPUTE_90A:
        return 90; // treat as 90 for clamping purposes
    default:
        return std::nullopt;
    }
}

std::optional<CUjit_target> find_sm_target(std::string_view text)
{
	const std::string_view marker = "sm_";
	size_t search_pos = 0;
	while (search_pos < text.size()) {
		auto pos = text.find(marker, search_pos);
		if (pos == std::string_view::npos)
			break;
		size_t current = pos + marker.size();
		bool has_digit = false;
		int value = 0;
		while (current < text.size() &&
		       std::isdigit(static_cast<unsigned char>(text[current]))) {
			has_digit = true;
			value = value * 10 + (text[current] - '0');
			current++;
		}
		bool accelerated = false;
		if (current < text.size() &&
		    (text[current] == 'a' || text[current] == 'A')) {
			accelerated = true;
			current++;
		}
		search_pos = current;
		if (!has_digit)
			continue;
		if (auto target = to_jit_target(value, accelerated))
			return target;
	}
	return std::nullopt;
}

std::optional<CUjit_target> deduce_jit_target(const std::string &module_name,
					      const std::string &ptx_text)
{
	if (auto from_name = find_sm_target(module_name))
		return from_name;
	return find_sm_target(ptx_text);
}
} // namespace

} // namespace bpftime::attach

namespace bpftime::attach
{
fatbin_record::~fatbin_record()
{
}
fatbin_record::ptx_in_module::~ptx_in_module()
{
	CUDA_DRIVER_CHECK_NO_EXCEPTION(cuModuleUnload(this->module_ptr),
				       "Unable to unload module");
}

bool fatbin_record::find_and_fill_variable_info(void *ptr,
						const char *symbol_name)
{
	for (const auto &ptx : ptxs) {
		CUdeviceptr dptr;
		size_t size;
		auto err = cuModuleGetGlobal(&dptr, &size, ptx->module_ptr,
					     symbol_name);
		if (err == CUDA_SUCCESS) {
			variable_addr_to_symbol[ptr] =
				variable_info{ .symbol_name =
						       std::string(symbol_name),
					       .ptr = dptr,
					       .size = size,
					       .ptx = ptx.get() };
			return true;
		} else if (err == CUDA_ERROR_NOT_FOUND) {
			continue;
		} else {
			SPDLOG_ERROR("Unable to lookup symbol: {}", (int)err);
			return false;
		}
	}
	return false;
}
bool fatbin_record::find_and_fill_function_info(void *ptr,
						const char *symbol_name)
{
	for (const auto &ptx : ptxs) {
		CUfunction func;
		auto err = cuModuleGetFunction(&func, ptx->module_ptr,
					       symbol_name);
		if (err == CUDA_SUCCESS) {
			function_addr_to_symbol[ptr] =
				kernel_info{ .symbol_name =
						     std::string(symbol_name),
					     .func = func,
					     .ptx = ptx.get() };
			return true;
		} else if (err == CUDA_ERROR_NOT_FOUND) {
			continue;
		} else {
			SPDLOG_ERROR("Unable to lookup function: {}", (int)err);
			return false;
		}
	}
	return false;
}

void fatbin_record::try_loading_ptxs(class nv_attach_impl &impl)
{
    if (ptx_loaded)
        return;
    SPDLOG_INFO("Loading & patching current fatbin..");
    auto patched_ptx = *impl.hack_fatbin(original_ptx);

    for (const auto &[name, ptx] : patched_ptx) {
        CUmodule module;
        SPDLOG_INFO("Loading module: {}", name);
        // Work on a mutable copy so we can rewrite .target if needed
        std::string ptx_text = ptx;
        char error_buf[8192]{}, info_buf[8192]{};
        std::array<CUjit_option, 8> options;
        std::array<void *, 8> option_values;
        size_t option_count = 0;
        options[option_count] = CU_JIT_INFO_LOG_BUFFER;
        option_values[option_count++] = (void *)info_buf;
        options[option_count] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        option_values[option_count++] = reinterpret_cast<void *>(
            static_cast<uintptr_t>(sizeof(info_buf)));
        options[option_count] = CU_JIT_ERROR_LOG_BUFFER;
        option_values[option_count++] = (void *)error_buf;
        options[option_count] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
        option_values[option_count++] = reinterpret_cast<void *>(
            static_cast<uintptr_t>(sizeof(error_buf)));
        options[option_count] = CU_JIT_FALLBACK_STRATEGY;
        option_values[option_count++] = reinterpret_cast<void *>(
            static_cast<uintptr_t>(CU_PREFER_PTX));
        // Compute and clamp target between module hint, PTX .target and device capability
        auto name_target = deduce_jit_target(name, ptx_text);
        auto ctx_target = device_default_target();
        std::optional<int> name_sm = name_target ? sm_from_jit_target(*name_target) : std::nullopt;
        std::optional<int> dev_sm = ctx_target ? sm_from_jit_target(*ctx_target) : std::nullopt;
        // Parse PTX .target sm_XX specifically from PTX text
        auto parse_ptx_target_sm = [](const std::string &text) -> std::optional<int> {
            size_t pos = 0;
            while (true) {
                pos = text.find(".target", pos);
                if (pos == std::string::npos)
                    return std::nullopt;
                // search until end of line
                size_t eol = text.find('\n', pos);
                size_t line_end = (eol == std::string::npos) ? text.size() : eol;
                size_t smpos = text.find("sm_", pos);
                if (smpos != std::string::npos && smpos < line_end) {
                    size_t cur = smpos + 3;
                    int value = 0;
                    bool has_digit = false;
                    while (cur < line_end && std::isdigit(static_cast<unsigned char>(text[cur]))) {
                        has_digit = true;
                        value = value * 10 + (text[cur] - '0');
                        ++cur;
                    }
                    if (has_digit)
                        return value;
                }
                // continue searching on next line
                pos = (line_end == text.size()) ? line_end : (line_end + 1);
            }
        };
        auto ptx_sm = parse_ptx_target_sm(ptx_text);

        std::optional<int> chosen_sm;
        if (dev_sm)
            chosen_sm = dev_sm;
        if (name_sm)
            chosen_sm = chosen_sm ? std::min(*chosen_sm, *name_sm) : name_sm;
        if (ptx_sm)
            chosen_sm = chosen_sm ? std::min(*chosen_sm, *ptx_sm) : ptx_sm;

        if (dev_sm || name_sm || ptx_sm) {
            int sm_to_use = chosen_sm.value_or(dev_sm.value_or(ptx_sm.value_or(0)));
            auto jit_target = to_jit_target(sm_to_use, false);
            if (jit_target) {
                // Normalize all PTX .target lines to the chosen target to avoid ptxas mismatch
                {
                    size_t pos = 0;
                    int rewrites = 0;
                    while (true) {
                        pos = ptx_text.find(".target", pos);
                        if (pos == std::string::npos)
                            break;
                        size_t eol = ptx_text.find('\n', pos);
                        size_t line_end = (eol == std::string::npos) ? ptx_text.size() : eol;
                        size_t smpos = ptx_text.find("sm_", pos);
                        if (smpos != std::string::npos && smpos < line_end) {
                            size_t cur = smpos + 3;
                            size_t digits_start = cur;
                            while (cur < line_end && std::isdigit(static_cast<unsigned char>(ptx_text[cur]))) {
                                ++cur;
                            }
                            auto before = std::string(ptx_text.substr(digits_start, cur - digits_start));
                            ptx_text.replace(digits_start, cur - digits_start, std::to_string(sm_to_use));
                            rewrites++;
                            SPDLOG_DEBUG("Rewriting PTX .target sm_{} -> sm_{} for {} (line at {})", before, sm_to_use, name, (int)pos);
                        }
                        pos = (line_end == ptx_text.size()) ? line_end : (line_end + 1);
                    }
                    if (rewrites > 0) {
                        SPDLOG_INFO("Rewrote {} PTX .target line(s) to sm_{} for {}", rewrites, sm_to_use, name);
                    }
                }
                unsigned int target_value = static_cast<unsigned int>(*jit_target);
                options[option_count] = CU_JIT_TARGET;
                option_values[option_count++] = reinterpret_cast<void *>(
                    static_cast<uintptr_t>(target_value));
                SPDLOG_DEBUG("Using CU_JIT_TARGET={} (sm_{}) for {} (name_sm={}, ptx_sm={}, dev_sm={})",
                             target_value, sm_to_use, name,
                             name_sm ? *name_sm : -1,
                             ptx_sm ? *ptx_sm : -1,
                             dev_sm ? *dev_sm : -1);
            } else {
                // Fallback to context if mapping failed
                options[option_count] = CU_JIT_TARGET_FROM_CUCONTEXT;
                option_values[option_count++] = nullptr;
                SPDLOG_DEBUG("Using CU_JIT_TARGET_FROM_CUCONTEXT for {} (failed map sm_{})", name, sm_to_use);
            }
        } else {
            options[option_count] = CU_JIT_TARGET_FROM_CUCONTEXT;
            option_values[option_count++] = nullptr;
            SPDLOG_DEBUG("Using CU_JIT_TARGET_FROM_CUCONTEXT for {} (no target hints)", name);
        }

        // Lower optimization level to improve robustness across older drivers
        options[option_count] = CU_JIT_OPTIMIZATION_LEVEL;
        option_values[option_count++] = reinterpret_cast<void *>(static_cast<uintptr_t>(0));

        if (auto err = cuModuleLoadDataEx(&module, ptx_text.data(),
                                 option_count, options.data(),
                                 option_values.data());
            err != CUDA_SUCCESS) {
            SPDLOG_ERROR("Unable to compile module {}: {}", name,
                         (int)err);
            SPDLOG_ERROR("Info: {}", info_buf);
            SPDLOG_ERROR("Error: {}", error_buf);
            throw std::runtime_error("Unable to compile module");
        }
		CUdeviceptr const_data_ptr, map_basic_info_ptr;
		size_t const_data_size, map_basic_info_size;
		SPDLOG_INFO("Copying trampoline data to device");
		CUDA_DRIVER_CHECK_EXCEPTION(
			cuModuleGetGlobal(&const_data_ptr, &const_data_size,
					  module, "constData"),
			"Unable to get pointer of constData");
		CUDA_DRIVER_CHECK_EXCEPTION(
			cuModuleGetGlobal(&map_basic_info_ptr,
					  &map_basic_info_size, module,
					  "map_info"),
			"Unable to get pointer of map_info");
		CUDA_DRIVER_CHECK_EXCEPTION(
			cuMemcpyHtoD(const_data_ptr, &impl.shared_mem_ptr,
				     const_data_size),
			"Unable to copy constData pointer to device");
		CUDA_DRIVER_CHECK_EXCEPTION(
			cuMemcpyHtoD(map_basic_info_ptr,
				     impl.map_basic_info->data(),
				     map_basic_info_size),
			"Unable to copy constData pointer to device");
		SPDLOG_INFO("Trampoline data copied");
		ptxs.emplace_back(
			std::make_unique<fatbin_record::ptx_in_module>(module));
		SPDLOG_INFO("Loaded module: {}", name);
	}
	ptx_loaded = true;
}

} // namespace bpftime::attach
