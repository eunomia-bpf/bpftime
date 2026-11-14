/* SPDX-License-Identifier: MIT */
#pragma once

#include <cstdint>

extern "C" {
int64_t bpftime_probe_read(uint64_t dst, int64_t size, uint64_t src, uint64_t,
			   uint64_t);
int64_t bpftime_probe_read_user(uint64_t dst, int64_t size, uint64_t src,
				uint64_t, uint64_t);
int64_t bpftime_probe_write_user(uint64_t dst, uint64_t src, int64_t size,
				 uint64_t, uint64_t);
int64_t bpf_probe_read_str(uint64_t dst, uint64_t size, uint64_t src, uint64_t,
			   uint64_t);
}
