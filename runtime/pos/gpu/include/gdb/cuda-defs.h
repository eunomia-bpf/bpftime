/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2015 NVIDIA Corporation
 * Written by CUDA-GDB team at NVIDIA <cudatools@nvidia.com>
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 */

#ifndef _CUDA_DEFS_H
#define _CUDA_DEFS_H 1


#include "cudadebugger.h"

/* Early declarations to avoid cyclic dependencies */

struct context_st;
struct contexts_st;
struct disasm_cache_st;
struct elf_image_st;
struct kernel_st;
struct module_st;
struct modules_st;
struct cuda_iterator_t;
struct cuda_exception_t;
struct regmap_st;
struct cuda_exception_st;

typedef struct context_st        *context_t;
typedef struct contexts_st       *contexts_t;
typedef struct disasm_cache_st   *disasm_cache_t;
typedef struct elf_image_st      *elf_image_t;
typedef struct kernel_st         *kernel_t;
typedef struct module_st         *module_t;
typedef struct modules_st        *modules_t;
typedef struct cuda_iterator_t   *cuda_iterator;
typedef struct regmap_st         *regmap_t;
typedef uint64_t                  cuda_clock_t;
typedef struct cuda_exception_st *cuda_exception_t;

extern const bool CACHED;
#endif
