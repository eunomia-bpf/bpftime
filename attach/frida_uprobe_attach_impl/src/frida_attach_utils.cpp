#include "frida_attach_utils.hpp"
#include "frida_uprobe_attach_impl.hpp"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <spdlog/spdlog.h>
#include <frida-gum.h>
#include <sys/stat.h>
#if __linux__
#include <sys/sysmacros.h>
#endif
#include <unistd.h>
#if __APPLE__
#include <libproc.h>
#endif
static std::string get_executable_path()
{
	char exec_path[PATH_MAX] = { 0 };

#if __linux__
	ssize_t len =
		readlink("/proc/self/exe", exec_path, sizeof(exec_path) - 1);
	if (len != -1) {
		exec_path[len] = '\0'; // Null-terminate the string
		SPDLOG_INFO("Executable path: {}", exec_path);
	} else {
		SPDLOG_ERROR("Error retrieving executable path: {}", errno);
	}
#elif __APPLE__
	pid_t pid = getpid();
	if (proc_pidpath(pid, exec_path, sizeof(exec_path)) > 0) {
		SPDLOG_INFO("Executable path: {}", exec_path);
	} else {
		SPDLOG_ERROR("Error retrieving executable path: {}", errno);
	}
#endif
	return exec_path;
}
namespace bpftime
{
namespace attach
{
namespace
{
std::string unescape_proc_path(std::string path)
{
	std::string result;
	result.reserve(path.size());
	for (size_t i = 0; i < path.size(); i++) {
		if (path[i] == '\\' && i + 3 < path.size() &&
		    path[i + 1] >= '0' && path[i + 1] <= '7' &&
		    path[i + 2] >= '0' && path[i + 2] <= '7' &&
		    path[i + 3] >= '0' && path[i + 3] <= '7') {
			result.push_back((char)((path[i + 1] - '0') * 64 +
					        (path[i + 2] - '0') * 8 +
					        path[i + 3] - '0'));
			i += 3;
		} else {
			result.push_back(path[i]);
		}
	}
	return result;
}
// Membership is checked against individual VMAs, not their outer bounds.
bool module_contains_address(std::string_view module_name, uintptr_t address);
} // namespace

std::optional<std::string>
resolve_mapped_module_path(const std::string_view &module_name)
{
#if __linux__
	std::string name(module_name);
	unsigned pid = 0;
	unsigned long long wanted_start = 0, wanted_end = 0;
	int consumed = 0;
	if (sscanf(name.c_str(), "/proc/%u/map_files/%llx-%llx%n", &pid,
		   &wanted_start, &wanted_end, &consumed) != 3 ||
	    consumed != static_cast<int>(name.size()))
		return name;
	if ((pid_t)pid != getpid() || wanted_start >= wanted_end)
		return {};

	std::ifstream maps("/proc/self/maps");
	std::string line;
	while (std::getline(maps, line)) {
		unsigned long long start, end, offset, inode;
		unsigned dev_major, dev_minor;
		char permissions[5] = {};
		int path_offset = 0;
		if (sscanf(line.c_str(),
			   "%llx-%llx %4s %llx %x:%x %llu %n", &start, &end,
			   permissions, &offset, &dev_major, &dev_minor, &inode,
			   &path_offset) != 7 ||
		    end <= wanted_start || start >= wanted_end)
			continue;
		std::string path = unescape_proc_path(line.substr(path_offset));
		struct stat st = {};
		if (path.empty() || path.front() != '/' ||
		    path.ends_with(" (deleted)") || stat(path.c_str(), &st) != 0 ||
		    (inode != 0 && st.st_ino != inode) ||
		    major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor)
			return {};
		return path;
	}
	return {};
#else
	return std::string(module_name);
#endif
}

void *
resolve_function_addr_by_module_offset(const std::string_view &module_name,
				       uintptr_t func_offset)
{
	auto exec_path = get_executable_path();
	std::error_code ec;
	// Resolve the executable alias the same way for the base and its VMAs.
	const std::string resolved_name =
		(module_name.empty() ||
		 std::filesystem::equivalent(module_name, exec_path, ec))
			? "" : std::string(module_name);
	void *module_base_addr = get_module_base_addr(resolved_name.c_str());
	if (!module_base_addr) {
		// It's not a bug, it might be attach to a unrelated process
		// when using the LD_PRELOAD
		SPDLOG_INFO("Failed to find module base address for {}",
			    module_name);
		return nullptr;
	}

	// Reject both integer overflow and unmapped gaps. Missing mappings fail
	// closed: an unchecked address must never reach Frida's trampoline writer.
	const auto base = reinterpret_cast<uintptr_t>(module_base_addr);
	if (func_offset > UINTPTR_MAX - base ||
	    !module_contains_address(resolved_name, base + func_offset)) {
		SPDLOG_ERROR("Module offset 0x{:x} is not mapped in {}",
			     func_offset, module_name);
		return nullptr;
	}
	return reinterpret_cast<void *>(base + func_offset);
}

void *find_function_addr_by_name(const char *name)
{
	if (auto ptr = gum_find_function(name); ptr)
		return ptr;
	if (auto ptr = (void *)gum_module_find_export_by_name(nullptr, name);
	    ptr)
		return ptr;
	return nullptr;
}

void *get_module_base_addr(const char *module_name)
{
	gum_module_load(module_name, nullptr);
	return (void *)gum_module_find_base_address(module_name);
}

namespace {
struct ExtentAccumulator {
	uintptr_t begin = UINTPTR_MAX;
	uintptr_t end = 0;
};

gboolean accumulate_range(const GumRangeDetails *details, gpointer user_data)
{
	auto *acc = static_cast<ExtentAccumulator *>(user_data);
	const auto base = (uintptr_t)details->range->base_address;
	const auto range_end = base + (uintptr_t)details->range->size;
	if (base < acc->begin)
		acc->begin = base;
	if (range_end > acc->end)
		acc->end = range_end;
	return TRUE;
}

struct AddressLookup {
	uintptr_t address;
	bool found = false;
};

gboolean find_mapped_address(const GumRangeDetails *details, gpointer user_data)
{
	auto *lookup = static_cast<AddressLookup *>(user_data);
	const auto start = (uintptr_t)details->range->base_address;
	if (lookup->address >= start &&
	    lookup->address - start < (uintptr_t)details->range->size) {
		lookup->found = true;
		return FALSE;
	}
	return TRUE;
}

bool module_contains_address(std::string_view module_name, uintptr_t address)
{
	const std::string name(module_name);
	AddressLookup lookup{address};
	gum_module_enumerate_ranges(name.c_str(), GUM_PAGE_READ,
				    find_mapped_address, &lookup);
	return lookup.found;
}
} // namespace

// Returns the span covering the module's mapped ranges (possibly with gaps),
// or 0 when the ranges cannot be determined.
uintptr_t module_mapped_extent(const std::string_view &module_name)
{
	const std::string name(module_name);
	ExtentAccumulator acc;
	gum_module_enumerate_ranges(name.c_str(), GUM_PAGE_READ,
				    accumulate_range, &acc);
	if (acc.end <= acc.begin)
		return 0;
	return acc.end - acc.begin;
}
void *find_module_export_by_name(const char *module_name,
				 const char *symbol_name)
{
	return (void *)(uintptr_t)gum_module_find_export_by_name(module_name,
								 symbol_name);
}
int from_cb_idx_to_attach_type(int idx)
{
	switch (idx) {
	case ATTACH_UPROBE_INDEX:
		return ATTACH_UPROBE;
	case ATTACH_UPROBE_OVERRIDE_INDEX:
		return ATTACH_UPROBE_OVERRIDE;
	case ATTACH_URETPROBE_INDEX:
		return ATTACH_URETPROBE;
	default:
		SPDLOG_ERROR("Unreachable branch reached!");
		return -1;
	}
	return 0;
}
} // namespace attach
} // namespace bpftime

extern "C" uint64_t bpftime_get_func_ret(uint64_t ctx, uint64_t *value,
					 uint64_t, uint64_t, uint64_t)
{
	GumInvocationContext *gum_ctx =
		gum_interceptor_get_current_invocation();
	if (gum_ctx == NULL) {
		return -EOPNOTSUPP;
	}
	// ignore ctx;
	*value = (uint64_t)gum_invocation_context_get_return_value(gum_ctx);
	return 0;
}

extern "C" uint64_t bpftime_get_func_arg(uint64_t ctx, uint32_t n,
					 uint64_t *value, uint64_t, uint64_t)
{
	GumInvocationContext *gum_ctx =
		gum_interceptor_get_current_invocation();
	if (gum_ctx == NULL) {
		return -EINVAL;
	}
	// ignore ctx;
	*value = (uint64_t)gum_cpu_context_get_nth_argument(
		gum_ctx->cpu_context, n);
	return 0;
}

extern "C" uint64_t bpftime_get_retval(uint64_t, uint64_t, uint64_t, uint64_t,
				       uint64_t)
{
	GumInvocationContext *gum_ctx =
		gum_interceptor_get_current_invocation();
	if (gum_ctx == NULL) {
		return -EOPNOTSUPP;
	}
	return (uintptr_t)gum_invocation_context_get_return_value(gum_ctx);
}
