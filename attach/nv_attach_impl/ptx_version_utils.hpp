#ifndef BPFTIME_PTX_VERSION_UTILS_HPP
#define BPFTIME_PTX_VERSION_UTILS_HPP

// Header-only so it can be shared between nv_attach_impl and the standalone
// nv_attach_impl_ptx_compiler shared library, which do not link together.

#include <cctype>
#include <cstring>
#include <optional>
#include <string>
#include <utility>

namespace bpftime
{
namespace attach
{
namespace ptx_version
{

// Returns [begin, end) of the numeric value following the first `.version`
// directive, or nullopt if there is no such directive.
inline std::optional<std::pair<std::size_t, std::size_t> >
locate(const std::string &ptx)
{
	auto pos = ptx.find(".version");
	if (pos == std::string::npos)
		return std::nullopt;
	pos += strlen(".version");
	while (pos < ptx.size() &&
	       std::isspace(static_cast<unsigned char>(ptx[pos]))) {
		pos++;
	}
	auto start = pos;
	while (pos < ptx.size() &&
	       (std::isdigit(static_cast<unsigned char>(ptx[pos])) ||
		ptx[pos] == '.')) {
		pos++;
	}
	if (start == pos)
		return std::nullopt;
	return std::make_pair(start, pos);
}

inline std::string rewrite(std::string ptx, const std::string &version)
{
	auto range = locate(ptx);
	if (!range)
		return ptx;
	ptx.replace(range->first, range->second - range->first, version);
	return ptx;
}

// Parses the ISA version accepted by the compiler out of an nvPTXCompiler
// error log, e.g. "PTX ISA version 9.2 is not supported, current version is
// '8.5'".
inline std::optional<std::string>
supported_version_from_error_log(const std::string &error_log)
{
	const char *patterns[] = { "current version is '",
				   "current version is \"" };
	for (const char *pattern : patterns) {
		auto pos = error_log.find(pattern);
		if (pos == std::string::npos)
			continue;
		pos += strlen(pattern);
		auto end = pos;
		while (end < error_log.size() &&
		       (std::isdigit(
				static_cast<unsigned char>(error_log[end])) ||
			error_log[end] == '.')) {
			end++;
		}
		if (end > pos)
			return error_log.substr(pos, end - pos);
	}
	return std::nullopt;
}

// Downgrades the declared `.version` when it exceeds the ISA version reported
// by nvPTXCompilerGetVersion. Generated PTX (and the bundled trampoline) may
// come from a newer toolkit than the nvPTXCompiler available at runtime.
inline std::string clamp_to(std::string ptx, unsigned int max_major,
			    unsigned int max_minor)
{
	auto range = locate(ptx);
	if (!range)
		return ptx;
	const auto declared =
		ptx.substr(range->first, range->second - range->first);
	auto dot = declared.find('.');
	if (dot == std::string::npos)
		return ptx;
	unsigned int major = 0, minor = 0;
	try {
		major = std::stoul(declared.substr(0, dot));
		minor = std::stoul(declared.substr(dot + 1));
	} catch (...) {
		return ptx;
	}
	if (major < max_major || (major == max_major && minor <= max_minor))
		return ptx;
	ptx.replace(range->first, range->second - range->first,
		    std::to_string(max_major) + "." + std::to_string(max_minor));
	return ptx;
}

} // namespace ptx_version
} // namespace attach
} // namespace bpftime

#endif
