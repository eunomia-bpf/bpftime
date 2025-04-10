/* Branch trace support for GDB, the GNU debugger.

   Copyright (C) 2013 Free Software Foundation, Inc.

   Contributed by Intel Corp. <markus.t.metzger@intel.com>.

   This file is part of GDB.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#ifndef BTRACE_COMMON_H
#define BTRACE_COMMON_H

/* Branch tracing (btrace) is a per-thread control-flow execution trace of the
   inferior.  For presentation purposes, the branch trace is represented as a
   list of sequential control-flow blocks, one such list per thread.  */

#ifdef GDBSERVER
#  include "server.h"
#else
#  include "defs.h"
#endif

#include "vec.h"

/* A branch trace block.

   This represents a block of sequential control-flow.  Adjacent blocks will be
   connected via calls, returns, or jumps.  The latter can be direct or
   indirect, conditional or unconditional.  Branches can further be
   asynchronous, e.g. interrupts.  */
struct btrace_block
{
  /* The address of the first byte of the first instruction in the block.  */
  CORE_ADDR begin;

  /* The address of the first byte of the last instruction in the block.  */
  CORE_ADDR end;
};

/* Branch trace is represented as a vector of branch trace blocks starting with
   the most recent block.  */
typedef struct btrace_block btrace_block_s;

/* Define functions operating on a vector of branch trace blocks.  */
DEF_VEC_O (btrace_block_s);

/* Target specific branch trace information.  */
struct btrace_target_info;

/* Enumeration of btrace read types.  */

enum btrace_read_type
{
  /* Send all available trace.  */
  btrace_read_all,

  /* Send all available trace, if it changed.  */
  btrace_read_new
};

#endif /* BTRACE_COMMON_H */
