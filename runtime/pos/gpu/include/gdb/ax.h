/* Definitions for expressions designed to be executed on the agent
   Copyright (C) 1998-2013 Free Software Foundation, Inc.

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

#ifndef AGENTEXPR_H
#define AGENTEXPR_H

#include "doublest.h"		/* For DOUBLEST.  */
#include "vec.h"

/* It's sometimes useful to be able to debug programs that you can't
   really stop for more than a fraction of a second.  To this end, the
   user can specify a tracepoint (like a breakpoint, but you don't
   stop at it), and specify a bunch of expressions to record the
   values of when that tracepoint is reached.  As the program runs,
   GDB collects the values.  At any point (possibly while values are
   still being collected), the user can display the collected values.

   This is used with remote debugging; we don't really support it on
   native configurations.

   This means that expressions are being evaluated by the remote agent,
   which doesn't have any access to the symbol table information, and
   needs to be small and simple.

   The agent_expr routines and datatypes are a bytecode language
   designed to be executed by the agent.  Agent expressions work in
   terms of fixed-width values, operators, memory references, and
   register references.  You can evaluate a agent expression just given
   a bunch of memory and register values to sniff at; you don't need
   any symbolic information like variable names, types, etc.

   GDB translates source expressions, whose meaning depends on
   symbolic information, into agent bytecode expressions, whose meaning
   is independent of symbolic information.  This means the agent can
   evaluate them on the fly without reference to data only available
   to the host GDB.  */


/* Different kinds of flaws an agent expression might have, as
   detected by ax_reqs.  */
enum agent_flaws
  {
    agent_flaw_none = 0,	/* code is good */

    /* There is an invalid instruction in the stream.  */
    agent_flaw_bad_instruction,

    /* There is an incomplete instruction at the end of the expression.  */
    agent_flaw_incomplete_instruction,

    /* ax_reqs was unable to prove that every jump target is to a
       valid offset.  Valid offsets are within the bounds of the
       expression, and to a valid instruction boundary.  */
    agent_flaw_bad_jump,

    /* ax_reqs was unable to prove to its satisfaction that, for each
       jump target location, the stack will have the same height whether
       that location is reached via a jump or by straight execution.  */
    agent_flaw_height_mismatch,

    /* ax_reqs was unable to prove that every instruction following
       an unconditional jump was the target of some other jump.  */
    agent_flaw_hole
  };

/* Agent expression data structures.  */

/* The type of an element of the agent expression stack.
   The bytecode operation indicates which element we should access;
   the value itself has no typing information.  GDB generates all
   bytecode streams, so we don't have to worry about type errors.  */

union agent_val
  {
    LONGEST l;
    DOUBLEST d;
  };

/* A buffer containing a agent expression.  */
struct agent_expr
  {
    /* The bytes of the expression.  */
    unsigned char *buf;

    /* The number of bytecode in the expression.  */
    int len;

    /* Allocated space available currently.  */
    int size;

    /* The target architecture assumed to be in effect.  */
    struct gdbarch *gdbarch;

    /* The address to which the expression applies.  */
    CORE_ADDR scope;

    /* If the following is not equal to agent_flaw_none, the rest of the
       information in this structure is suspect.  */
    enum agent_flaws flaw;

    /* Number of elements left on stack at end; may be negative if expr
       only consumes elements.  */
    int final_height;

    /* Maximum and minimum stack height, relative to initial height.  */
    int max_height, min_height;

    /* Largest `ref' or `const' opcode used, in bits.  Zero means the
       expression has no such instructions.  */
    int max_data_size;

    /* Bit vector of registers needed.  Register R is needed iff

       reg_mask[R / 8] & (1 << (R % 8))

       is non-zero.  Note!  You may not assume that this bitmask is long
       enough to hold bits for all the registers of the machine; the
       agent expression code has no idea how many registers the machine
       has.  However, the bitmask is reg_mask_len bytes long, so the
       valid register numbers run from 0 to reg_mask_len * 8 - 1.

       Also note that this mask may contain registers that are needed
       for the original collection expression to work, but that are
       not referenced by any bytecode.  This could, for example, occur
       when collecting a local variable allocated to a register; the
       compiler sets the mask bit and skips generating a bytecode whose
       result is going to be discarded anyway.
    */
    int reg_mask_len;
    unsigned char *reg_mask;
  };

/* Pointer to an agent_expr structure.  */
typedef struct agent_expr *agent_expr_p;

/* Vector of pointers to agent expressions.  */
DEF_VEC_P (agent_expr_p);

/* The actual values of the various bytecode operations.  */

enum agent_op
  {
#define DEFOP(NAME, SIZE, DATA_SIZE, CONSUMED, PRODUCED, VALUE)  \
    aop_ ## NAME = VALUE,
#include "ax.def"
#undef DEFOP
    aop_last
  };



/* Functions for building expressions.  */

/* Allocate a new, empty agent expression.  */
extern struct agent_expr *new_agent_expr (struct gdbarch *, CORE_ADDR);

/* Free a agent expression.  */
extern void free_agent_expr (struct agent_expr *);
extern struct cleanup *make_cleanup_free_agent_expr (struct agent_expr *);

/* Append a simple operator OP to EXPR.  */
extern void ax_simple (struct agent_expr *EXPR, enum agent_op OP);

/* Append a pick operator to EXPR.  DEPTH is the stack item to pick,
   with 0 being top of stack.  */
extern void ax_pick (struct agent_expr *EXPR, int DEPTH);

/* Append the floating-point prefix, for the next bytecode.  */
#define ax_float(EXPR) (ax_simple ((EXPR), aop_float))

/* Append a sign-extension instruction to EXPR, to extend an N-bit value.  */
extern void ax_ext (struct agent_expr *EXPR, int N);

/* Append a zero-extension instruction to EXPR, to extend an N-bit value.  */
extern void ax_zero_ext (struct agent_expr *EXPR, int N);

/* Append a trace_quick instruction to EXPR, to record N bytes.  */
extern void ax_trace_quick (struct agent_expr *EXPR, int N);

/* Append a goto op to EXPR.  OP is the actual op (must be aop_goto or
   aop_if_goto).  We assume we don't know the target offset yet,
   because it's probably a forward branch, so we leave space in EXPR
   for the target, and return the offset in EXPR of that space, so we
   can backpatch it once we do know the target offset.  Use ax_label
   to do the backpatching.  */
extern int ax_goto (struct agent_expr *EXPR, enum agent_op OP);

/* Suppose a given call to ax_goto returns some value PATCH.  When you
   know the offset TARGET that goto should jump to, call
   ax_label (EXPR, PATCH, TARGET)
   to patch TARGET into the ax_goto instruction.  */
extern void ax_label (struct agent_expr *EXPR, int patch, int target);

/* Assemble code to push a constant on the stack.  */
extern void ax_const_l (struct agent_expr *EXPR, LONGEST l);
extern void ax_const_d (struct agent_expr *EXPR, LONGEST d);

/* Assemble code to push the value of register number REG on the
   stack.  */
extern void ax_reg (struct agent_expr *EXPR, int REG);

/* Add the given register to the register mask of the expression.  */
extern void ax_reg_mask (struct agent_expr *ax, int reg);

/* Assemble code to operate on a trace state variable.  */
extern void ax_tsv (struct agent_expr *expr, enum agent_op op, int num);

/* Append a string to the bytecode stream.  */
extern void ax_string (struct agent_expr *x, const char *str, int slen);


/* Functions for printing out expressions, and otherwise debugging
   things.  */

/* Disassemble the expression EXPR, writing to F.  */
extern void ax_print (struct ui_file *f, struct agent_expr * EXPR);

/* An entry in the opcode map.  */
struct aop_map
  {

    /* The name of the opcode.  Null means that this entry is not a
       valid opcode --- a hole in the opcode space.  */
    const char *name;

    /* All opcodes take no operands from the bytecode stream, or take
       unsigned integers of various sizes.  If this is a positive number
       n, then the opcode is followed by an n-byte operand, which should
       be printed as an unsigned integer.  If this is zero, then the
       opcode takes no operands from the bytecode stream.

       If we get more complicated opcodes in the future, don't add other
       magic values of this; that's a crock.  Add an `enum encoding'
       field to this, or something like that.  */
    int op_size;

    /* The size of the data operated upon, in bits, for bytecodes that
       care about that (ref and const).  Zero for all others.  */
    int data_size;

    /* Number of stack elements consumed, and number produced.  */
    int consumed, produced;
  };

/* Map of the bytecodes, indexed by bytecode number.  */
extern struct aop_map aop_map[];

/* Given an agent expression AX, analyze and update its requirements.  */

extern void ax_reqs (struct agent_expr *ax);

#endif /* AGENTEXPR_H */
