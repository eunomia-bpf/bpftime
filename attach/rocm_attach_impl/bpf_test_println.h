// SPDX-License-Identifier: MIT
#ifndef BPFTIME_ROCM_TEST_PRINTLN_H
#define BPFTIME_ROCM_TEST_PRINTLN_H
// ROCm device helper: pass a valid NUL-terminated device string.
static long (*const bpf_test_println)(const char *) = (void *)506;
#endif
