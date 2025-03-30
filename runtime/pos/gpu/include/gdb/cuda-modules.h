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

#ifndef _CUDA_MODULES_H
#define _CUDA_MODULES_H 1

#include "cuda-defs.h"

module_t    module_new    (context_t context, uint64_t module_id,
                           void *elf_image, uint64_t elf_image_size);
void        module_delete (module_t module);
void        module_print  (module_t module);

uint64_t    module_get_id         (module_t module);
context_t   module_get_context    (module_t module);
elf_image_t module_get_elf_image  (module_t module);

modules_t  modules_new    (void);
void       modules_delete (modules_t modules);
void       modules_add    (modules_t modules, module_t module);
void       modules_remove (modules_t modules, module_t module);
void       modules_print  (modules_t modules);

module_t   modules_find_module_by_id      (modules_t modules, uint64_t module_id);
module_t   modules_find_module_by_address (modules_t modules, CORE_ADDR addr);

#endif

