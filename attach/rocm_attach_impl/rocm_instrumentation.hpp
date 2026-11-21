// SPDX-License-Identifier: MIT
#pragma once
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>
#include <luthier/Common/DynamicLibrary.h>
namespace bpftime::attach
{
// Capture pre-interception tables; no HSA calls until the first dispatch.
void capture_rocm_tables(const HsaApiTable &,
			 const luthier::DynamicLibrary &) noexcept;
// This function owns forwarding of the entire batch, exactly once per packet.
void instrument_rocm_packets(const hsa_queue_t &, const void *, uint64_t,
			     hsa_amd_queue_intercept_packet_writer) noexcept;
void invalidate_rocm_executable(hsa_executable_t) noexcept;
void shutdown_rocm_instrumentation() noexcept;
void suspend_rocm_instrumentation() noexcept;
} // namespace bpftime::attach
