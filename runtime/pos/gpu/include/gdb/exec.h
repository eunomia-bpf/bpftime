/* Work with executable files, for GDB, the GNU debugger.

   Copyright (C) 2003-2013 Free Software Foundation, Inc.

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

#ifndef EXEC_H
#define EXEC_H

#include "target.h"
#include "progspace.h"
#include "memrange.h"

struct target_section;
struct target_ops;
struct bfd;

extern struct target_ops exec_ops;

#define exec_bfd current_program_space->ebfd
#define exec_bfd_mtime current_program_space->ebfd_mtime
#define exec_filename current_program_space->pspace_exec_filename

/* Builds a section table, given args BFD, SECTABLE_PTR, SECEND_PTR.
   Returns 0 if OK, 1 on error.  */

extern int build_section_table (struct bfd *, struct target_section **,
				struct target_section **);

/* Resize the section table held by TABLE, by NUM_ADDED.  Returns the
   old size.  */

extern int resize_section_table (struct target_section_table *, int);

/* Appends all read-only memory ranges found in the target section
   table defined by SECTIONS and SECTIONS_END, starting at (and
   intersected with) MEMADDR for LEN bytes.  Returns the augmented
   VEC.  */

extern VEC(mem_range_s) *
  section_table_available_memory (VEC(mem_range_s) *ranges,
				  CORE_ADDR memaddr, ULONGEST len,
				  struct target_section *sections,
				  struct target_section *sections_end);

/* Read or write from mappable sections of BFD executable files.

   Request to transfer up to LEN 8-bit bytes of the target sections
   defined by SECTIONS and SECTIONS_END.  The OFFSET specifies the
   starting address.
   If SECTION_NAME is not NULL, only access sections with that same
   name.

   Return the number of bytes actually transfered, or zero when no
   data is available for the requested range.

   This function is intended to be used from target_xfer_partial
   implementations.  See target_read and target_write for more
   information.

   One, and only one, of readbuf or writebuf must be non-NULL.  */

extern int section_table_xfer_memory_partial (gdb_byte *, const gdb_byte *,
					      ULONGEST, LONGEST,
					      struct target_section *,
					      struct target_section *,
					      const char *);

/* Set the loaded address of a section.  */
extern void exec_set_section_address (const char *, int, CORE_ADDR);

/* Remove all target sections taken from ABFD.  */

extern void remove_target_sections (void *key, bfd *abfd);

/* Add the sections array defined by [SECTIONS..SECTIONS_END[ to the
   current set of target sections.  */

extern void add_target_sections (void *key,
				 struct target_section *sections,
				 struct target_section *sections_end);

/* Prints info about all sections defined in the TABLE.  ABFD is
   special cased --- it's filename is omitted; if it is the executable
   file, its entry point is printed.  */

extern void print_section_info (struct target_section_table *table,
				bfd *abfd);

extern void exec_close (void);

#endif
