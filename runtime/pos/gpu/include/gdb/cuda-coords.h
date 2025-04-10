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

#ifndef _CUDA_COORDS_H
#define _CUDA_COORDS_H 1

#include "defs.h"
#include "cuda-defs.h"
#include "cudadebugger.h"

typedef struct {
  bool valid;
  uint64_t kernelId;
  uint64_t gridId;
  CuDim3 blockIdx;
  CuDim3 threadIdx;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t ln;
} cuda_coords_t;

typedef struct {
  bool valid;
  ptid_t ptid;
  cuda_coords_t coords;
} cuda_focus_t;

typedef enum {
  CK_EXACT_LOGICAL,
  CK_EXACT_PHYSICAL,
  CK_CLOSEST_LOGICAL,
  CK_CLOSEST_PHYSICAL,
  CK_LOWEST_PHYSICAL,
  CK_LOWEST_LOGICAL,
  CK_NEXT_PHYSICAL,
  CK_NEXT_LOGICAL,
  CK_MAX
} cuda_coords_kind_t;

typedef enum {
  CUDA_INVALID  = ~0U,
  CUDA_WILDCARD = CUDA_INVALID - 1,
  CUDA_CURRENT  = CUDA_INVALID - 2,
} cuda_coords_special_value_t;

typedef enum {
  CUDA_SELECT_ALL   = 0x000,
  CUDA_SELECT_VALID = 0x001,
  CUDA_SELECT_BKPT  = 0x002,
  CUDA_SELECT_EXCPT = 0x004,
  CUDA_SELECT_SNGL  = 0x008,
} cuda_select_t;

#define CUDA_INVALID_COORDS ((cuda_coords_t)                             \
            { false, CUDA_INVALID, CUDA_INVALID,                         \
            { CUDA_INVALID, CUDA_INVALID, CUDA_INVALID },                \
            { CUDA_INVALID, CUDA_INVALID, CUDA_INVALID },                \
            CUDA_INVALID, CUDA_INVALID, CUDA_INVALID, CUDA_INVALID })

#define CUDA_WILDCARD_COORDS ((cuda_coords_t)                            \
            { true, CUDA_WILDCARD, CUDA_WILDCARD,                        \
            { CUDA_WILDCARD, CUDA_WILDCARD, CUDA_WILDCARD },             \
            { CUDA_WILDCARD, CUDA_WILDCARD, CUDA_WILDCARD },             \
            CUDA_WILDCARD, CUDA_WILDCARD, CUDA_WILDCARD, CUDA_WILDCARD })

#define CUDA_COORD_IS_SPECIAL(x)                                         \
  ((x) == CUDA_INVALID || (x) == CUDA_WILDCARD || (x) == CUDA_CURRENT)

/*Current Focus */
bool  cuda_focus_is_device (void);
void  cuda_coords_update_current (bool breakpoint_hit, bool exception_hit);
void  cuda_coords_invalidate_current (void);
void  cuda_coords_reset_current (void);
int   cuda_coords_set_current (cuda_coords_t *c);
int   cuda_coords_set_current_logical (uint64_t kernelId, uint64_t gridId, CuDim3 blockIdx, CuDim3 threadIdx);
int   cuda_coords_set_current_physical (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln);
int   cuda_coords_get_current (cuda_coords_t *c);
int   cuda_coords_get_current_logical (uint64_t *kernelId, uint64_t *gridId, CuDim3 *blockIdx, CuDim3 *threadIdx);
int   cuda_coords_get_current_physical (uint32_t *dev, uint32_t *sm, uint32_t *wp, uint32_t *ln);
bool  cuda_coords_is_current (cuda_coords_t *c);
bool  cuda_coords_is_current_logical (cuda_coords_t *c);
void  cuda_coords_find_valid (cuda_coords_t wished, cuda_coords_t found[CK_MAX], cuda_select_t select_mask);
void  cuda_focus_init (cuda_focus_t *focus);
void  cuda_focus_save (cuda_focus_t *focus);
void  cuda_focus_restore (cuda_focus_t *focus);
void  cuda_coords_increment_block (cuda_coords_t *c, CuDim3 grid_dim);
void  cuda_coords_increment_thread (cuda_coords_t *c, CuDim3 grid_dim, CuDim3 block_dim);

uint32_t cuda_current_device (void);
uint32_t cuda_current_sm (void);
uint32_t cuda_current_warp (void);
uint32_t cuda_current_lane (void);
kernel_t cuda_current_kernel (void);

/*Coordinates Manipulation */
bool  cuda_coords_equal (cuda_coords_t *c1, cuda_coords_t *c2);
int   cuda_coords_compare_logical (cuda_coords_t *c1, cuda_coords_t *c2);
int   cuda_coords_compare_physical (cuda_coords_t *c1, cuda_coords_t *c2);
void  cuda_coords_to_fancy_string (cuda_coords_t *c, char *string, uint32_t size);
void  cuda_string_to_coords (bool is_mask, char *block_repr, char *thread_repr, cuda_coords_t *current, cuda_coords_t *mask);
int   cuda_parse_thread(char **ptok, char **pcuda_block_mask_repr, char** pcuda_thread_mask_repr, int *pFoundEntry);
int   cuda_coords_complete_logical (cuda_coords_t *c);
int   cuda_coords_complete_physical (cuda_coords_t *c);
void  cuda_coords_evaluate_current (cuda_coords_t *coords, bool use_wildcards);
void  cuda_coords_check_fully_defined (cuda_coords_t *coords, bool accept_invalid, bool accept_current, bool accept_wildcards);

void  cuda_print_message_focus (bool switching);

#endif

