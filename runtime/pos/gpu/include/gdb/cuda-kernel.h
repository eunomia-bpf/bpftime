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


#ifndef _CUDA_KERNEL_H
#define _CUDA_KERNEL_H 1

#include "cuda-defs.h"

uint64_t            kernel_get_id                       (kernel_t kernel);
uint32_t            kernel_get_dev_id                   (kernel_t kernel);
uint64_t            kernel_get_grid_id                  (kernel_t kernel);
kernel_t            kernel_get_parent                   (kernel_t kernel);
kernel_t            kernel_get_children                 (kernel_t kernel);
kernel_t            kernel_get_sibling                  (kernel_t kernel);
const char*         kernel_get_name                     (kernel_t kernel);
const char*         kernel_get_args                     (kernel_t kernel);
uint64_t            kernel_get_virt_code_base           (kernel_t kernel);
context_t           kernel_get_context                  (kernel_t kernel);
module_t            kernel_get_module                   (kernel_t kernel);
CuDim3              kernel_get_grid_dim                 (kernel_t kernel);
CuDim3              kernel_get_block_dim                (kernel_t kernel);
const char*         kernel_get_dimensions               (kernel_t kernel);
CUDBGKernelType     kernel_get_type                     (kernel_t kernel);
CUDBGGridStatus     kernel_get_status                   (kernel_t kernel);
CUDBGKernelOrigin   kernel_get_origin                   (kernel_t kernel);
uint32_t            kernel_get_depth                    (kernel_t kernel);
uint32_t            kernel_get_num_children             (kernel_t kernel);
bool                kernel_has_launched                 (kernel_t kernel);
bool                kernel_is_present                   (kernel_t kernel);

void                kernel_invalidate         (kernel_t kernel);
uint32_t            kernel_compute_sms_mask   (kernel_t kernel);
void                kernel_print              (kernel_t kernel);
void                kernel_flush_disasm_cache (kernel_t kernel);
const char*         kernel_disassemble        (kernel_t kernel, uint64_t pc,
                                           uint32_t *inst_size);

void      kernels_start_kernel     (uint32_t dev_id, uint64_t grid_id,
                                    uint64_t virt_code_base,
                                    uint64_t context_id, uint64_t module_id,
                                    CuDim3 grid_dim, CuDim3 block_dim,
                                    CUDBGKernelType type,
                                    uint64_t parent_grid_id,
                                    CUDBGKernelOrigin origin);
void      kernels_terminate_kernel  (kernel_t kernel);
void      kernels_terminate_module  (module_t module);
void      kernels_update_terminated (void);
void      kernels_update_args       (void);
void      kernels_print             (void);
kernel_t  kernels_get_first_kernel  (void);
kernel_t  kernels_get_next_kernel   (kernel_t kernel);
kernel_t  kernels_find_kernel_by_grid_id   (uint32_t dev_id, uint64_t grid_id);
kernel_t  kernels_find_kernel_by_kernel_id (uint64_t kernel_id);

uint64_t  cuda_latest_launched_kernel_id (void);

#endif
