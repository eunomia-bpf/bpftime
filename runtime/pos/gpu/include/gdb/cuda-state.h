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

#ifndef _CUDA_STATE_H
#define _CUDA_STATE_H 1

#include "cuda-defs.h"
#include "cuda-tdep.h"

/* System State */
void     cuda_system_initialize                   (void);
void     cuda_system_finalize                     (void);
uint32_t cuda_system_get_num_devices              (void);
uint32_t cuda_system_get_num_kernels              (void);
uint32_t cuda_system_get_num_present_kernels      (void);
void     cuda_system_resolve_breakpoints          (int bp_number_from);
void     cuda_system_cleanup_breakpoints          (void);
void     cuda_system_cleanup_contexts             (void);
bool     cuda_system_is_broken                    (cuda_clock_t);
uint32_t cuda_system_get_suspended_devices_mask   (void);
void     cuda_system_flush_disasm_cache           (void);

void     cuda_system_set_device_spec    (uint32_t, uint32_t, uint32_t,
                                         uint32_t, uint32_t, char *, char *);

context_t cuda_system_find_context_by_addr     (CORE_ADDR addr);

/* Device State */
const char* device_get_device_name         (uint32_t dev_id);
const char* device_get_device_type         (uint32_t dev_id);
const char* device_get_sm_type             (uint32_t dev_id);
uint32_t    device_get_num_sms             (uint32_t dev_id);
uint32_t    device_get_num_warps           (uint32_t dev_id);
uint32_t    device_get_num_lanes           (uint32_t dev_id);
uint32_t    device_get_num_registers       (uint32_t dev_id);
uint32_t    device_get_num_predicates      (uint32_t dev_id);
uint32_t    device_get_num_kernels         (uint32_t dev_id);
uint32_t    device_get_pci_bus_id          (uint32_t dev_id);
uint32_t    device_get_pci_dev_id          (uint32_t dev_id);

bool        device_is_valid                (uint32_t dev_id);
bool        device_is_any_context_present  (uint32_t dev_id);
bool        device_is_active_context       (uint32_t dev_id, context_t context);
bool        device_has_exception           (uint32_t dev_id);
uint64_t    device_get_active_sms_mask     (uint32_t dev_id);
contexts_t  device_get_contexts            (uint32_t dev_id);

context_t   device_find_context_by_id      (uint32_t dev_id, uint64_t context_id);
context_t   device_find_context_by_addr    (uint32_t dev_id, CORE_ADDR addr);

void        device_print      (uint32_t dev_id);
void        device_resume     (uint32_t dev_id);
void        device_suspend    (uint32_t dev_id);
void        device_invalidate (uint32_t dev_id);

/* SM State */
bool        sm_is_valid                    (uint32_t dev_id, uint32_t sm_id);
bool        sm_has_exception               (uint32_t dev_id, uint32_t sm_id);
uint64_t    sm_get_valid_warps_mask        (uint32_t dev_id, uint32_t sm_id);
uint64_t    sm_get_broken_warps_mask       (uint32_t dev_id, uint32_t sm_id);

/* Warp State */
bool     warp_is_valid                 (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
bool     warp_is_broken                (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
bool     warp_has_error_pc             (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
kernel_t warp_get_kernel               (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
CuDim3   warp_get_block_idx            (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
uint64_t warp_get_grid_id              (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
uint32_t warp_get_valid_lanes_mask     (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
uint32_t warp_get_active_lanes_mask    (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
uint32_t warp_get_divergent_lanes_mask (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
uint32_t warp_get_lowest_active_lane   (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
uint64_t warp_get_active_pc            (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
uint64_t warp_get_active_virtual_pc    (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
uint64_t warp_get_error_pc             (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
cuda_clock_t warp_get_timestamp        (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
void     warp_set_grid_id              (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint64_t grid_id);
void     warp_set_block_idx            (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, CuDim3 *block_idx);

bool     warp_single_step              (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id,   uint64_t *single_stepped_warp_mask);
bool     warps_resume_until            (uint32_t dev_id, uint32_t sm_id, uint64_t wp_mask, uint64_t pc);

/* Lane State */
bool             lane_is_valid       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);
bool             lane_is_active      (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);
bool             lane_is_divergent   (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);
uint64_t         lane_get_pc         (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);
uint64_t         lane_get_virtual_pc (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);
CuDim3           lane_get_thread_idx (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);
CUDBGException_t lane_get_exception  (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);
uint32_t         lane_get_register   (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, uint32_t regno);
uint32_t         lane_get_cc_register (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);
bool             lane_get_predicate  (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, uint32_t predicate);
void             lane_set_register   (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, uint32_t regno, uint32_t value);
void             lane_set_predicate  (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, uint32_t predicate, bool value);
void             lane_set_cc_register (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, uint32_t value);
int32_t          lane_get_call_depth (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);
int32_t          lane_get_syscall_call_depth (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);
uint64_t         lane_get_virtual_return_address (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, int32_t level);
cuda_clock_t     lane_get_timestamp (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id,uint32_t ln_id);
uint64_t         lane_get_memcheck_error_address (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);
ptxStorageKind   lane_get_memcheck_error_address_segment (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);
void             lane_set_thread_idx (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, CuDim3 *thread_idx);
#endif
