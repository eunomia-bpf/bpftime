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

#ifndef _CUDA_ELF_IMAGES_H
#define _CUDA_ELF_IMAGES_H 1

#include "symtab.h"
#include "objfiles.h"
#include "cudadebugger.h"
#include "cuda-defs.h"

extern elf_image_t elf_image_chain;

#define CUDA_ALL_ELF_IMAGES(E)                                                \
        for ((E) = elf_image_chain;                                           \
             (E) && cuda_elf_image_get_next (E);                              \
             (E) = cuda_elf_image_get_next (E))

#define CUDA_ALL_LOADED_ELF_IMAGES(E)                                         \
        CUDA_ALL_ELF_IMAGES (E)                                               \
          if (cuda_elf_image_is_loaded (E))

elf_image_t cuda_elf_image_new (void *image, uint64_t size, module_t module);
void        cuda_elf_image_delete (elf_image_t elf_image);

void *           cuda_elf_image_get_image        (elf_image_t elf_image);
struct objfile * cuda_elf_image_get_objfile      (elf_image_t elf_image);
uint64_t         cuda_elf_image_get_size         (elf_image_t elf_image);
module_t         cuda_elf_image_get_module       (elf_image_t elf_image);
elf_image_t      cuda_elf_image_get_next         (elf_image_t elf_image);

bool             cuda_elf_image_is_loaded        (elf_image_t elf_image);
bool             cuda_elf_image_uses_abi         (elf_image_t elf_image);
bool             cuda_elf_image_is_system        (elf_image_t elf_image);

void             cuda_elf_image_save             (elf_image_t elf_image, void *image);
void             cuda_elf_image_load             (elf_image_t elf_image, bool is_system);
void             cuda_elf_image_unload           (elf_image_t elf_image);

bool             cuda_elf_image_contains_address (elf_image_t elf_image, CORE_ADDR addr);
void             cuda_elf_image_resolve_breakpoints (elf_image_t elf_image);

#endif
