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

#ifndef BTRACE_H
#define BTRACE_H

/* Branch tracing (btrace) is a per-thread control-flow execution trace of the
   inferior.  For presentation purposes, the branch trace is represented as a
   list of sequential control-flow blocks, one such list per thread.  */

#include "btrace-common.h"

struct thread_info;

/* A branch trace instruction.

   This represents a single instruction in a branch trace.  */
struct btrace_inst
{
  /* The address of this instruction.  */
  CORE_ADDR pc;
};

/* A branch trace function.

   This represents a function segment in a branch trace, i.e. a consecutive
   number of instructions belonging to the same function.  */
struct btrace_func
{
  /* The full and minimal symbol for the function.  One of them may be NULL.  */
  struct minimal_symbol *msym;
  struct symbol *sym;

  /* The source line range of this function segment (both inclusive).  */
  int lbegin, lend;

  /* The instruction number range in the instruction trace corresponding
     to this function segment (both inclusive).  */
  unsigned int ibegin, iend;
};

/* Branch trace may also be represented as a vector of:

   - branch trace instructions starting with the oldest instruction.
   - branch trace functions starting with the oldest function.  */
typedef struct btrace_inst btrace_inst_s;
typedef struct btrace_func btrace_func_s;

/* Define functions operating on branch trace vectors.  */
DEF_VEC_O (btrace_inst_s);
DEF_VEC_O (btrace_func_s);

/* Branch trace iteration state for "record instruction-history".  */
struct btrace_insn_iterator
{
  /* The instruction index range from begin (inclusive) to end (exclusive)
     that has been covered last time.
     If end < begin, the branch trace has just been updated.  */
  unsigned int begin;
  unsigned int end;
};

/* Branch trace iteration state for "record function-call-history".  */
struct btrace_func_iterator
{
  /* The function index range from begin (inclusive) to end (exclusive)
     that has been covered last time.
     If end < begin, the branch trace has just been updated.  */
  unsigned int begin;
  unsigned int end;
};

/* Branch trace information per thread.

   This represents the branch trace configuration as well as the entry point
   into the branch trace data.  For the latter, it also contains the index into
   an array of branch trace blocks used for iterating though the branch trace
   blocks of a thread.  */
struct btrace_thread_info
{
  /* The target branch trace information for this thread.

     This contains the branch trace configuration as well as any
     target-specific information necessary for implementing branch tracing on
     the underlying architecture.  */
  struct btrace_target_info *target;

  /* The current branch trace for this thread.  */
  VEC (btrace_block_s) *btrace;
  VEC (btrace_inst_s) *itrace;
  VEC (btrace_func_s) *ftrace;

  /* The instruction history iterator.  */
  struct btrace_insn_iterator insn_iterator;

  /* The function call history iterator.  */
  struct btrace_func_iterator func_iterator;
};

/* Enable branch tracing for a thread.  */
extern void btrace_enable (struct thread_info *tp);

/* Disable branch tracing for a thread.
   This will also delete the current branch trace data.  */
extern void btrace_disable (struct thread_info *);

/* Disable branch tracing for a thread during teardown.
   This is similar to btrace_disable, except that it will use
   target_teardown_btrace instead of target_disable_btrace.  */
extern void btrace_teardown (struct thread_info *);

/* Fetch the branch trace for a single thread.  */
extern void btrace_fetch (struct thread_info *);

/* Clear the branch trace for a single thread.  */
extern void btrace_clear (struct thread_info *);

/* Clear the branch trace for all threads when an object file goes away.  */
extern void btrace_free_objfile (struct objfile *);

/* Parse a branch trace xml document into a block vector.  */
extern VEC (btrace_block_s) *parse_xml_btrace (const char*);

#endif /* BTRACE_H */
