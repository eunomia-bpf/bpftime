/* Program and address space management, for GDB, the GNU debugger.

   Copyright (C) 2009-2013 Free Software Foundation, Inc.

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


#ifndef PROGSPACE_H
#define PROGSPACE_H

#include "target.h"
#include "vec.h"
#include "gdb_vecs.h"
#include "registry.h"

struct target_ops;
struct bfd;
struct objfile;
struct inferior;
struct exec;
struct address_space;
struct program_space_data;

typedef struct so_list *so_list_ptr;
DEF_VEC_P (so_list_ptr);

/* A program space represents a symbolic view of an address space.
   Roughly speaking, it holds all the data associated with a
   non-running-yet program (main executable, main symbols), and when
   an inferior is running and is bound to it, includes the list of its
   mapped in shared libraries.

   In the traditional debugging scenario, there's a 1-1 correspondence
   among program spaces, inferiors and address spaces, like so:

     pspace1 (prog1) <--> inf1(pid1) <--> aspace1

   In the case of debugging more than one traditional unix process or
   program, we still have:

     |-----------------+------------+---------|
     | pspace1 (prog1) | inf1(pid1) | aspace1 |
     |----------------------------------------|
     | pspace2 (prog1) | no inf yet | aspace2 |
     |-----------------+------------+---------|
     | pspace3 (prog2) | inf2(pid2) | aspace3 |
     |-----------------+------------+---------|

   In the former example, if inf1 forks (and GDB stays attached to
   both processes), the new child will have its own program and
   address spaces.  Like so:

     |-----------------+------------+---------|
     | pspace1 (prog1) | inf1(pid1) | aspace1 |
     |-----------------+------------+---------|
     | pspace2 (prog1) | inf2(pid2) | aspace2 |
     |-----------------+------------+---------|

   However, had inf1 from the latter case vforked instead, it would
   share the program and address spaces with its parent, until it
   execs or exits, like so:

     |-----------------+------------+---------|
     | pspace1 (prog1) | inf1(pid1) | aspace1 |
     |                 | inf2(pid2) |         |
     |-----------------+------------+---------|

   When the vfork child execs, it is finally given new program and
   address spaces.

     |-----------------+------------+---------|
     | pspace1 (prog1) | inf1(pid1) | aspace1 |
     |-----------------+------------+---------|
     | pspace2 (prog1) | inf2(pid2) | aspace2 |
     |-----------------+------------+---------|

   There are targets where the OS (if any) doesn't provide memory
   management or VM protection, where all inferiors share the same
   address space --- e.g. uClinux.  GDB models this by having all
   inferiors share the same address space, but, giving each its own
   program space, like so:

     |-----------------+------------+---------|
     | pspace1 (prog1) | inf1(pid1) |         |
     |-----------------+------------+         |
     | pspace2 (prog1) | inf2(pid2) | aspace1 |
     |-----------------+------------+         |
     | pspace3 (prog2) | inf3(pid3) |         |
     |-----------------+------------+---------|

   The address space sharing matters for run control and breakpoints
   management.  E.g., did we just hit a known breakpoint that we need
   to step over?  Is this breakpoint a duplicate of this other one, or
   do I need to insert a trap?

   Then, there are targets where all symbols look the same for all
   inferiors, although each has its own address space, as e.g.,
   Ericsson DICOS.  In such case, the model is:

     |---------+------------+---------|
     |         | inf1(pid1) | aspace1 |
     |         +------------+---------|
     | pspace  | inf2(pid2) | aspace2 |
     |         +------------+---------|
     |         | inf3(pid3) | aspace3 |
     |---------+------------+---------|

   Note however, that the DICOS debug API takes care of making GDB
   believe that breakpoints are "global".  That is, although each
   process does have its own private copy of data symbols (just like a
   bunch of forks), to the breakpoints module, all processes share a
   single address space, so all breakpoints set at the same address
   are duplicates of each other, even breakpoints set in the data
   space (e.g., call dummy breakpoints placed on stack).  This allows
   a simplification in the spaces implementation: we avoid caring for
   a many-many links between address and program spaces.  Either
   there's a single address space bound to the program space
   (traditional unix/uClinux), or, in the DICOS case, the address
   space bound to the program space is mostly ignored.  */

/* The program space structure.  */

struct program_space
  {
    /* Pointer to next in linked list.  */
    struct program_space *next;

    /* Unique ID number.  */
    int num;

    /* The main executable loaded into this program space.  This is
       managed by the exec target.  */

    /* The BFD handle for the main executable.  */
    bfd *ebfd;
    /* The last-modified time, from when the exec was brought in.  */
    long ebfd_mtime;
    /* Similar to bfd_get_filename (exec_bfd) but in original form given
       by user, without symbolic links and pathname resolved.
       It needs to be freed by xfree.  It is not NULL iff EBFD is not NULL.  */
    char *pspace_exec_filename;

    /* The address space attached to this program space.  More than one
       program space may be bound to the same address space.  In the
       traditional unix-like debugging scenario, this will usually
       match the address space bound to the inferior, and is mostly
       used by the breakpoints module for address matches.  If the
       target shares a program space for all inferiors and breakpoints
       are global, then this field is ignored (we don't currently
       support inferiors sharing a program space if the target doesn't
       make breakpoints global).  */
    struct address_space *aspace;

    /* True if this program space's section offsets don't yet represent
       the final offsets of the "live" address space (that is, the
       section addresses still require the relocation offsets to be
       applied, and hence we can't trust the section addresses for
       anything that pokes at live memory).  E.g., for qOffsets
       targets, or for PIE executables, until we connect and ask the
       target for the final relocation offsets, the symbols we've used
       to set breakpoints point at the wrong addresses.  */
    int executing_startup;

    /* True if no breakpoints should be inserted in this program
       space.  */
    int breakpoints_not_allowed;

    /* The object file that the main symbol table was loaded from
       (e.g. the argument to the "symbol-file" or "file" command).  */
    struct objfile *symfile_object_file;

    /* All known objfiles are kept in a linked list.  This points to
       the head of this list.  */
    struct objfile *objfiles;

    /* The set of target sections matching the sections mapped into
       this program space.  Managed by both exec_ops and solib.c.  */
    struct target_section_table target_sections;

    /* List of shared objects mapped into this space.  Managed by
       solib.c.  */
    struct so_list *so_list;

    /* Number of calls to solib_add.  */
    unsigned solib_add_generation;

    /* When an solib is added, it is also added to this vector.  This
       is so we can properly report solib changes to the user.  */
    VEC (so_list_ptr) *added_solibs;

    /* When an solib is removed, its name is added to this vector.
       This is so we can properly report solib changes to the user.  */
    VEC (char_ptr) *deleted_solibs;

    /* Per pspace data-pointers required by other GDB modules.  */
    REGISTRY_FIELDS;
  };

/* The object file that the main symbol table was loaded from (e.g. the
   argument to the "symbol-file" or "file" command).  */

#define symfile_objfile current_program_space->symfile_object_file

/* All known objfiles are kept in a linked list.  This points to the
   root of this list.  */
#define object_files current_program_space->objfiles

/* The set of target sections matching the sections mapped into the
   current program space.  */
#define current_target_sections (&current_program_space->target_sections)

/* The list of all program spaces.  There's always at least one.  */
extern struct program_space *program_spaces;

/* The current program space.  This is always non-null.  */
extern struct program_space *current_program_space;

#define ALL_PSPACES(pspace) \
  for ((pspace) = program_spaces; (pspace) != NULL; (pspace) = (pspace)->next)

/* Add a new empty program space, and assign ASPACE to it.  Returns the
   pointer to the new object.  */
extern struct program_space *add_program_space (struct address_space *aspace);

/* Release PSPACE and removes it from the pspace list.  */
extern void remove_program_space (struct program_space *pspace);

/* Returns the number of program spaces listed.  */
extern int number_of_program_spaces (void);

/* Copies program space SRC to DEST.  Copies the main executable file,
   and the main symbol file.  Returns DEST.  */
extern struct program_space *clone_program_space (struct program_space *dest,
						struct program_space *src);

/* Save the current program space so that it may be restored by a later
   call to do_cleanups.  Returns the struct cleanup pointer needed for
   later doing the cleanup.  */
extern struct cleanup *save_current_program_space (void);

/* Sets PSPACE as the current program space.  This is usually used
   instead of set_current_space_and_thread when the current
   thread/inferior is not important for the operations that follow.
   E.g., when accessing the raw symbol tables.  If memory access is
   required, then you should use switch_to_program_space_and_thread.
   Otherwise, it is the caller's responsibility to make sure that the
   currently selected inferior/thread matches the selected program
   space.  */
extern void set_current_program_space (struct program_space *pspace);

/* Saves the current thread (may be null), frame and program space in
   the current cleanup chain.  */
extern struct cleanup *save_current_space_and_thread (void);

/* Switches full context to program space PSPACE.  Switches to the
   first thread found bound to PSPACE.  */
extern void switch_to_program_space_and_thread (struct program_space *pspace);

/* Create a new address space object, and add it to the list.  */
extern struct address_space *new_address_space (void);

/* Maybe create a new address space object, and add it to the list, or
   return a pointer to an existing address space, in case inferiors
   share an address space.  */
extern struct address_space *maybe_new_address_space (void);

/* Returns the integer address space id of ASPACE.  */
extern int address_space_num (struct address_space *aspace);

/* Update all program spaces matching to address spaces.  The user may
   have created several program spaces, and loaded executables into
   them before connecting to the target interface that will create the
   inferiors.  All that happens before GDB has a chance to know if the
   inferiors will share an address space or not.  Call this after
   having connected to the target interface and having fetched the
   target description, to fixup the program/address spaces
   mappings.  */
extern void update_address_spaces (void);

/* Prune away automatically added program spaces that aren't required
   anymore.  */
extern void prune_program_spaces (void);

/* Reset saved solib data at the start of an solib event.  This lets
   us properly collect the data when calling solib_add, so it can then
   later be printed.  */
extern void clear_program_space_solib_cache (struct program_space *);

/* Keep a registry of per-pspace data-pointers required by other GDB
   modules.  */

DECLARE_REGISTRY (program_space);

#endif
