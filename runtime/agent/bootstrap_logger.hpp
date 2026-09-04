// Configure diagnostics before CUDA registration or shared-memory initialization.
// The agent later applies the loader's runtime configuration as before.
#pragma once
#include "bpftime_logger.hpp"
#include <cstdlib>
#include <mutex>

namespace bpftime {
inline void initialize_agent_bootstrap_logger()
{
	static std::once_flag once;
	std::call_once(once, [] {
		const char *target = std::getenv("BPFTIME_LOG_OUTPUT");
		bpftime_set_logger(target != nullptr ? target : "");
	});
}
} // namespace bpftime
