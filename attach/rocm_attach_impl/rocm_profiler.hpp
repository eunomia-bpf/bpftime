// SPDX-License-Identifier: MIT
#pragma once

#include <rocprofiler-sdk/registration.h>

namespace bpftime::attach
{
// Callbacks must not invoke HIP/HSA: ROCprofiler runs tool initialization
// while the runtimes are still registering their API tables.
rocprofiler_tool_configure_result_t *
configure_rocm_profiler(uint32_t version, rocprofiler_client_id_t *client,
			bool (*initialize_agent)() noexcept,
			void (*finalize_agent)() noexcept) noexcept;
} // namespace bpftime::attach
