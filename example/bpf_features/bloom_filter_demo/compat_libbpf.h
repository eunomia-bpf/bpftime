// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
#pragma once

#if __has_include(<bpf/libbpf.h>)
#include <bpf/libbpf.h>
#elif __has_include("libbpf.h")
#include "libbpf.h"
#elif __has_include(<libbpf.h>)
#include <libbpf.h>
#else
#error "libbpf.h not found. Ensure headers are available under build/libbpf/bpf or bootstrap include paths."
#endif
