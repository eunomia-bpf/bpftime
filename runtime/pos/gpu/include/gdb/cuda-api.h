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

#ifndef _CUDA_API_H
#define _CUDA_API_H 1

#include "cudadebugger.h"

typedef enum {
  CUDA_ATTACH_STATE_NOT_STARTED,
  CUDA_ATTACH_STATE_IN_PROGRESS,
  CUDA_ATTACH_STATE_APP_READY,
  CUDA_ATTACH_STATE_COMPLETE,
  CUDA_ATTACH_STATE_DETACHING,
  CUDA_ATTACH_STATE_DETACH_COMPLETE
} cuda_attach_state_t;

typedef enum {
  CUDA_API_STATE_UNINITIALIZED,
  CUDA_API_STATE_INITIALIZED,
} cuda_api_state_t;

/* Initialization */
void cuda_api_handle_initialization_error (CUDBGResult res);
void cuda_api_handle_get_api_error (CUDBGResult res);
void cuda_api_handle_finalize_api_error (CUDBGResult res);
void cuda_api_set_api (CUDBGAPI api);
int  cuda_api_initialize (void);
void cuda_api_initialize_attach_stub (void);
void cuda_api_finalize (void);
void cuda_api_clear_state (void);
cuda_api_state_t cuda_api_get_state (void);

/* Attach support */
void cuda_api_set_attach_state (cuda_attach_state_t state);
bool cuda_api_attach_or_detach_in_progress (void);
cuda_attach_state_t cuda_api_get_attach_state (void);
void cuda_api_request_cleanup_on_detach (uint32_t resumeAppFlag);

/* Device Execution Control */
void cuda_api_suspend_device (uint32_t dev);
void cuda_api_resume_device (uint32_t dev);
bool cuda_api_resume_warps_until_pc (uint32_t dev, uint32_t sm, uint64_t warp_mask, uint64_t virt_pc);
bool cuda_api_single_step_warp (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *warp_mask);

/* Breakpoints */
bool cuda_api_set_breakpoint (uint32_t dev, uint64_t addr);
bool cuda_api_unset_breakpoint (uint32_t dev, uint64_t addr);
void cuda_api_get_adjusted_code_address (uint32_t dev, uint64_t addr, uint64_t *adjusted_addr, CUDBGAdjAddrAction adj_action);

/* Device State Inspection */
void cuda_api_read_grid_id (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *grid_id);
void cuda_api_read_block_idx (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *blockIdx);
void cuda_api_read_thread_idx (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CuDim3 *threadIdx);
void cuda_api_read_broken_warps (uint32_t dev, uint32_t sm, uint64_t *brokenWarpsMask);
void cuda_api_read_valid_warps (uint32_t dev, uint32_t sm, uint64_t *valid_warps);
void cuda_api_read_valid_lanes (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *valid_lanes);
void cuda_api_read_active_lanes (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *active_lanes);
void cuda_api_read_code_memory (uint32_t dev, uint64_t addr, void *buf, uint32_t sz);
void cuda_api_read_const_memory (uint32_t dev, uint64_t addr, void *buf, uint32_t sz);
void cuda_api_read_generic_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t sz);
bool cuda_api_read_pinned_memory (uint64_t addr, void *buf, uint32_t sz);
void cuda_api_read_param_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t sz);
void cuda_api_read_shared_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t sz);
void cuda_api_read_texture_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t id, uint32_t dim, uint32_t *coords, void *buf, uint32_t sz);
void cuda_api_read_texture_memory_bindless (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t tex_symtab_index, uint32_t dim, uint32_t *coords, void *buf, uint32_t sz);
void cuda_api_read_local_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t sz);
void cuda_api_read_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t regno, uint32_t *val);
void cuda_api_read_predicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t predicates_size, uint32_t *predicates);
void cuda_api_read_cc_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t *val);
void cuda_api_read_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc);
void cuda_api_read_virtual_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc);
void cuda_api_read_lane_exception (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CUDBGException_t *exception);
void cuda_api_read_call_depth (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, int32_t *depth);
void cuda_api_read_syscall_call_depth (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, int32_t *depth);
void cuda_api_read_virtual_return_address (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, int32_t level, uint64_t *ra);
void cuda_api_read_device_exception_state (uint32_t dev, uint64_t *exceptionSMMask);
void cuda_api_read_error_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *pc, bool* valid);
void cuda_api_read_warp_state (uint32_t dev, uint32_t sm, uint32_t wp, CUDBGWarpState *state);
void cuda_api_read_register_range (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t idx, uint32_t count, uint32_t *regs);
void cuda_api_read_global_memory (uint64_t addr, void *buf, uint32_t buf_size);
void cuda_api_write_global_memory (uint64_t addr, const void *buf, uint32_t buf_size);
void cuda_api_get_managed_memory_region_info (uint64_t start_addr, CUDBGMemoryInfo *meminfo, uint32_t entries_count, uint32_t *entries_written);

/* Device State Alteration */
void cuda_api_write_generic_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t sz);
bool cuda_api_write_pinned_memory (uint64_t addr, const void *buf, uint32_t sz);
void cuda_api_write_param_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t sz);
void cuda_api_write_shared_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t sz);
void cuda_api_write_local_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t sz);
void cuda_api_write_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t regno, uint32_t val);
void cuda_api_write_predicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t predicates_size, const uint32_t *predicates);
void cuda_api_write_cc_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t val);

/* Grid Properties */
void cuda_api_get_grid_dim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *grid_dim);
void cuda_api_get_block_dim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *block_dim);
void cuda_api_get_tid (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *tid);
void cuda_api_get_elf_image (uint32_t dev, uint64_t handle, bool relocated, void *elfImage, uint64_t size);
void cuda_api_get_blocking (uint32_t dev, uint32_t sm, uint32_t wp, bool *blocking);
void cuda_api_get_grid_status (uint32_t dev, uint64_t grid_id, CUDBGGridStatus *status);
void cuda_api_get_grid_info (uint32_t dev, uint64_t grid_id, CUDBGGridInfo *info);

/* Device Properties */
void cuda_api_get_device_name (uint32_t dev, char *buf, uint32_t sz);
void cuda_api_get_device_type (uint32_t dev, char *buf, uint32_t sz);
void cuda_api_get_sm_type (uint32_t dev, char *buf, uint32_t sz);
void cuda_api_get_num_devices (uint32_t *numDev);
void cuda_api_get_num_sms (uint32_t dev, uint32_t *numSMs);
void cuda_api_get_num_warps (uint32_t dev, uint32_t *numWarps);
void cuda_api_get_num_lanes (uint32_t dev, uint32_t *numLanes);
void cuda_api_get_num_registers (uint32_t dev, uint32_t *numRegs);
void cuda_api_get_num_predicates (uint32_t dev, uint32_t *numPredicates);
void cuda_api_get_device_pci_bus_info (uint32_t dev, uint32_t *pci_bus_id, uint32_t *pci_dev_id);

/* DWARF-related routines */
void cuda_api_disassemble (uint32_t dev, uint64_t addr, uint32_t *instSize, char *buf, uint32_t bufSize);
void cuda_api_is_device_code_address (uint64_t addr, bool *is_device_address);

/* Events */
void cuda_api_handle_set_callback_api_error (CUDBGResult res);
void cuda_api_set_notify_new_event_callback (CUDBGNotifyNewEventCallback callback);
void cuda_api_acknowledge_sync_events (void);
void cuda_api_get_next_sync_event (CUDBGEvent *event);
void cuda_api_get_next_async_event (CUDBGEvent *event);
void cuda_api_set_kernel_launch_notification_mode (CUDBGKernelLaunchNotifyMode mode);

/* Memcheck related */
void cuda_api_memcheck_read_error_address(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *address, ptxStorageKind *storage);
#endif

