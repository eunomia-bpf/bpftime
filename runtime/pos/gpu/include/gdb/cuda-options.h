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

#ifndef _CUDA_OPTIONS_H
#define _CUDA_OPTIONS_H 1

#include "cudadebugger.h"
#include "cuda-tdep.h"

void cuda_options_initialize (void);

bool cuda_options_debug_general (void);
bool cuda_options_debug_notifications (void);
bool cuda_options_debug_libcudbg (void);
bool cuda_options_debug_convenience_vars (void);
bool cuda_options_debug_strict (void);
bool cuda_options_memcheck (void);
bool cuda_options_coalescing (void);
bool cuda_options_break_on_launch_application(void);
bool cuda_options_break_on_launch_system (void);
bool cuda_options_disassemble_from_device_memory (void);
bool cuda_options_disassemble_from_elf_image (void);
bool cuda_options_hide_internal_frames (void);
void cuda_options_force_set_launch_notification_update (void);
unsigned int cuda_options_show_kernel_events_depth (void);
bool cuda_options_show_kernel_events_application (void);
bool cuda_options_show_kernel_events_system (void);
bool cuda_options_show_context_events (void);
bool cuda_options_launch_blocking (void);
bool cuda_options_thread_selection_logical (void);
bool cuda_options_thread_selection_physical (void);
bool cuda_options_api_failures_ignore (void);
bool cuda_options_api_failures_stop (void);
bool cuda_options_api_failures_hide (void);
bool cuda_options_api_failures_break_on_nonfatal(void);
void cuda_options_disable_break_on_launch (void);
bool cuda_options_notify_youngest (void);
bool cuda_options_notify_random (void);
bool cuda_options_software_preemption (void);
bool cuda_options_gpu_busy_check (void);
bool cuda_options_variable_value_cache_enabled (void);
bool cuda_options_statistics_collection_enabled (void);
bool cuda_options_value_extrapolation_enabled (void);
bool cuda_options_trace_domain_enabled (cuda_trace_domain_t);
bool cuda_options_single_stepping_optimizations_enabled (void);
/* Return GDB_SIGNAL_TRAP or GDB_SIGNAL_URG */
unsigned cuda_options_stop_signal (void);

/* Return true of BOL/KE breakpoints needs to be inserted */
bool cuda_options_auto_breakpoints_needed (void);

#endif

