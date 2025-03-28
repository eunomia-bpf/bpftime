/* Definitions for symbol file management in GDB.

   Copyright (C) 1992-2013 Free Software Foundation, Inc.

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

/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2013 NVIDIA Corporation
 * Modified from the original GDB file referenced above by the CUDA-GDB 
 * team at NVIDIA <cudatools@nvidia.com>.
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

#if !defined (OBJFILES_H)
#define OBJFILES_H

#include "gdb_obstack.h"	/* For obstack internals.  */
#include "symfile.h"		/* For struct psymbol_allocation_list.  */
#include "progspace.h"
#include "registry.h"

struct bcache;
struct htab;
struct symtab;
struct objfile_data;
struct cuda_regmap_table;

/* This structure maintains information on a per-objfile basis about the
   "entry point" of the objfile, and the scope within which the entry point
   exists.  It is possible that gdb will see more than one objfile that is
   executable, each with its own entry point.

   For example, for dynamically linked executables in SVR4, the dynamic linker
   code is contained within the shared C library, which is actually executable
   and is run by the kernel first when an exec is done of a user executable
   that is dynamically linked.  The dynamic linker within the shared C library
   then maps in the various program segments in the user executable and jumps
   to the user executable's recorded entry point, as if the call had been made
   directly by the kernel.

   The traditional gdb method of using this info was to use the
   recorded entry point to set the entry-file's lowpc and highpc from
   the debugging information, where these values are the starting
   address (inclusive) and ending address (exclusive) of the
   instruction space in the executable which correspond to the
   "startup file", i.e. crt0.o in most cases.  This file is assumed to
   be a startup file and frames with pc's inside it are treated as
   nonexistent.  Setting these variables is necessary so that
   backtraces do not fly off the bottom of the stack.

   NOTE: cagney/2003-09-09: It turns out that this "traditional"
   method doesn't work.  Corinna writes: ``It turns out that the call
   to test for "inside entry file" destroys a meaningful backtrace
   under some conditions.  E.g. the backtrace tests in the asm-source
   testcase are broken for some targets.  In this test the functions
   are all implemented as part of one file and the testcase is not
   necessarily linked with a start file (depending on the target).
   What happens is, that the first frame is printed normaly and
   following frames are treated as being inside the enttry file then.
   This way, only the #0 frame is printed in the backtrace output.''
   Ref "frame.c" "NOTE: vinschen/2003-04-01".

   Gdb also supports an alternate method to avoid running off the bottom
   of the stack.

   There are two frames that are "special", the frame for the function
   containing the process entry point, since it has no predecessor frame,
   and the frame for the function containing the user code entry point
   (the main() function), since all the predecessor frames are for the
   process startup code.  Since we have no guarantee that the linked
   in startup modules have any debugging information that gdb can use,
   we need to avoid following frame pointers back into frames that might
   have been built in the startup code, as we might get hopelessly 
   confused.  However, we almost always have debugging information
   available for main().

   These variables are used to save the range of PC values which are
   valid within the main() function and within the function containing
   the process entry point.  If we always consider the frame for
   main() as the outermost frame when debugging user code, and the
   frame for the process entry point function as the outermost frame
   when debugging startup code, then all we have to do is have
   DEPRECATED_FRAME_CHAIN_VALID return false whenever a frame's
   current PC is within the range specified by these variables.  In
   essence, we set "ceilings" in the frame chain beyond which we will
   not proceed when following the frame chain back up the stack.

   A nice side effect is that we can still debug startup code without
   running off the end of the frame chain, assuming that we have usable
   debugging information in the startup modules, and if we choose to not
   use the block at main, or can't find it for some reason, everything
   still works as before.  And if we have no startup code debugging
   information but we do have usable information for main(), backtraces
   from user code don't go wandering off into the startup code.  */

struct entry_info
  {
    /* The relocated value we should use for this objfile entry point.  */
    CORE_ADDR entry_point;

    /* Set to 1 iff ENTRY_POINT contains a valid value.  */
    unsigned entry_point_p : 1;
  };

/* Sections in an objfile.  The section offsets are stored in the
   OBJFILE.  */

struct obj_section
  {
    struct bfd_section *the_bfd_section;	/* BFD section pointer */

    /* Objfile this section is part of.  */
    struct objfile *objfile;

    /* True if this "overlay section" is mapped into an "overlay region".  */
    int ovly_mapped;
  };

/* Relocation offset applied to S.  */
#define obj_section_offset(s)						\
  (((s)->objfile->section_offsets)->offsets[(s)->the_bfd_section->index])

/* The memory address of section S (vma + offset).  */
#define obj_section_addr(s)				      		\
  (bfd_get_section_vma ((s)->objfile->obfd, s->the_bfd_section)		\
   + obj_section_offset (s))

/* The one-passed-the-end memory address of section S
   (vma + size + offset).  */
#define obj_section_endaddr(s)						\
  (bfd_get_section_vma ((s)->objfile->obfd, s->the_bfd_section)		\
   + bfd_get_section_size ((s)->the_bfd_section)			\
   + obj_section_offset (s))

/* The "objstats" structure provides a place for gdb to record some
   interesting information about its internal state at runtime, on a
   per objfile basis, such as information about the number of symbols
   read, size of string table (if any), etc.  */

struct objstats
  {
    int n_minsyms;		/* Number of minimal symbols read */
    int n_psyms;		/* Number of partial symbols read */
    int n_syms;			/* Number of full symbols read */
    int n_stabs;		/* Number of ".stabs" read (if applicable) */
    int n_types;		/* Number of types */
    int sz_strtab;		/* Size of stringtable, (if applicable) */
  };

#define OBJSTAT(objfile, expr) (objfile -> stats.expr)
#define OBJSTATS struct objstats stats
extern void print_objfile_statistics (void);
extern void print_symbol_bcache_statistics (void);

/* Number of entries in the minimal symbol hash table.  */
#define MINIMAL_SYMBOL_HASH_SIZE 2039

/* Some objfile data is hung off the BFD.  This enables sharing of the
   data across all objfiles using the BFD.  The data is stored in an
   instance of this structure, and associated with the BFD using the
   registry system.  */

struct objfile_per_bfd_storage
{
  /* The storage has an obstack of its own.  */

  struct obstack storage_obstack;
  
  /* Byte cache for file names.  */

  struct bcache *filename_cache;

  /* Byte cache for macros.  */
  struct bcache *macro_cache;
};

/* Master structure for keeping track of each file from which
   gdb reads symbols.  There are several ways these get allocated: 1.
   The main symbol file, symfile_objfile, set by the symbol-file command,
   2.  Additional symbol files added by the add-symbol-file command,
   3.  Shared library objfiles, added by ADD_SOLIB,  4.  symbol files
   for modules that were loaded when GDB attached to a remote system
   (see remote-vx.c).  */

struct objfile
  {

    /* All struct objfile's are chained together by their next pointers.
       The program space field "objfiles"  (frequently referenced via
       the macro "object_files") points to the first link in this
       chain.  */

    struct objfile *next;

    /* The object file's name, tilde-expanded and absolute.  This
       pointer is never NULL.  This does not have to be freed; it is
       guaranteed to have a lifetime at least as long as the objfile.  */

    char *name;

    CORE_ADDR addr_low;

    /* Some flag bits for this objfile.
       The values are defined by OBJF_*.  */

    unsigned short flags;

    /* The program space associated with this objfile.  */

    struct program_space *pspace;

    /* Each objfile points to a linked list of symtabs derived from this file,
       one symtab structure for each compilation unit (source file).  Each link
       in the symtab list contains a backpointer to this objfile.  */

    struct symtab *symtabs;

    /* Each objfile points to a linked list of partial symtabs derived from
       this file, one partial symtab structure for each compilation unit
       (source file).  */

    struct partial_symtab *psymtabs;

    /* Map addresses to the entries of PSYMTABS.  It would be more efficient to
       have a map per the whole process but ADDRMAP cannot selectively remove
       its items during FREE_OBJFILE.  This mapping is already present even for
       PARTIAL_SYMTABs which still have no corresponding full SYMTABs read.  */

    struct addrmap *psymtabs_addrmap;

    /* List of freed partial symtabs, available for re-use.  */

    struct partial_symtab *free_psymtabs;

    /* The object file's BFD.  Can be null if the objfile contains only
       minimal symbols, e.g. the run time common symbols for SunOS4.  */

    bfd *obfd;

    /* The per-BFD data.  Note that this is treated specially if OBFD
       is NULL.  */

    struct objfile_per_bfd_storage *per_bfd;

    /* The gdbarch associated with the BFD.  Note that this gdbarch is
       determined solely from BFD information, without looking at target
       information.  The gdbarch determined from a running target may
       differ from this e.g. with respect to register types and names.  */

    struct gdbarch *gdbarch;

    /* The modification timestamp of the object file, as of the last time
       we read its symbols.  */

    long mtime;

    /* Cached 32-bit CRC as computed by gnu_debuglink_crc32.  CRC32 is valid
       iff CRC32_P.  */
    unsigned long crc32;
    int crc32_p;

    /* Obstack to hold objects that should be freed when we load a new symbol
       table from this object file.  */

    struct obstack objfile_obstack; 

    /* A byte cache where we can stash arbitrary "chunks" of bytes that
       will not change.  */

    struct psymbol_bcache *psymbol_cache; /* Byte cache for partial syms.  */

    /* Hash table for mapping symbol names to demangled names.  Each
       entry in the hash table is actually two consecutive strings,
       both null-terminated; the first one is a mangled or linkage
       name, and the second is the demangled name or just a zero byte
       if the name doesn't demangle.  */
    struct htab *demangled_names_hash;

    /* Vectors of all partial symbols read in from file.  The actual data
       is stored in the objfile_obstack.  */

    struct psymbol_allocation_list global_psymbols;
    struct psymbol_allocation_list static_psymbols;

    /* Each file contains a pointer to an array of minimal symbols for all
       global symbols that are defined within the file.  The array is
       terminated by a "null symbol", one that has a NULL pointer for the
       name and a zero value for the address.  This makes it easy to walk
       through the array when passed a pointer to somewhere in the middle
       of it.  There is also a count of the number of symbols, which does
       not include the terminating null symbol.  The array itself, as well
       as all the data that it points to, should be allocated on the
       objfile_obstack for this file.  */

    struct minimal_symbol *msymbols;
    int minimal_symbol_count;

    /* This is a hash table used to index the minimal symbols by name.  */

    struct minimal_symbol *msymbol_hash[MINIMAL_SYMBOL_HASH_SIZE];

    /* This is a hash table used to index the minimal symbols by lowercase name.  */

    struct minimal_symbol *msymbol_lowercase_hash[MINIMAL_SYMBOL_HASH_SIZE];

    /* This hash table is used to index the minimal symbols by their
       demangled names.  */

    struct minimal_symbol *msymbol_demangled_hash[MINIMAL_SYMBOL_HASH_SIZE];

    /* Structure which keeps track of functions that manipulate objfile's
       of the same type as this objfile.  I.e. the function to read partial
       symbols for example.  Note that this structure is in statically
       allocated memory, and is shared by all objfiles that use the
       object module reader of this type.  */

    const struct sym_fns *sf;

    /* The per-objfile information about the entry point, the scope (file/func)
       containing the entry point, and the scope of the user's main() func.  */

    struct entry_info ei;

    /* Per objfile data-pointers required by other GDB modules.  */

    REGISTRY_FIELDS;

    /* Set of relocation offsets to apply to each section.
       The table is indexed by the_bfd_section->index, thus it is generally
       as large as the number of sections in the binary.
       The table is stored on the objfile_obstack.

       These offsets indicate that all symbols (including partial and
       minimal symbols) which have been read have been relocated by this
       much.  Symbols which are yet to be read need to be relocated by it.  */

    struct section_offsets *section_offsets;
    int num_sections;

    /* Indexes in the section_offsets array.  These are initialized by the
       *_symfile_offsets() family of functions (som_symfile_offsets,
       xcoff_symfile_offsets, default_symfile_offsets).  In theory they
       should correspond to the section indexes used by bfd for the
       current objfile.  The exception to this for the time being is the
       SOM version.  */

    int sect_index_text;
    int sect_index_data;
    int sect_index_bss;
    int sect_index_rodata;

    /* These pointers are used to locate the section table, which
       among other things, is used to map pc addresses into sections.
       SECTIONS points to the first entry in the table, and
       SECTIONS_END points to the first location past the last entry
       in the table.  The table is stored on the objfile_obstack.
       There is no particular order to the sections in this table, and it
       only contains sections we care about (e.g. non-empty, SEC_ALLOC).  */

    struct obj_section *sections, *sections_end;

    /* GDB allows to have debug symbols in separate object files.  This is
       used by .gnu_debuglink, ELF build id note and Mach-O OSO.
       Although this is a tree structure, GDB only support one level
       (ie a separate debug for a separate debug is not supported).  Note that
       separate debug object are in the main chain and therefore will be
       visited by ALL_OBJFILES & co iterators.  Separate debug objfile always
       has a non-nul separate_debug_objfile_backlink.  */

    /* Link to the first separate debug object, if any.  */
    struct objfile *separate_debug_objfile;

    /* If this is a separate debug object, this is used as a link to the
       actual executable objfile.  */
    struct objfile *separate_debug_objfile_backlink;

    /* If this is a separate debug object, this is a link to the next one
       for the same executable objfile.  */
    struct objfile *separate_debug_objfile_link;

    /* Place to stash various statistics about this objfile.  */
    OBJSTATS;

    /* A linked list of symbols created when reading template types or
       function templates.  These symbols are not stored in any symbol
       table, so we have to keep them here to relocate them
       properly.  */
    struct symbol *template_symbols;

    /* CUDA - cuda_objfile */
    int cuda_objfile;

    /* CUDA - regmap */
    struct cuda_regmap_table *cuda_regmap;

    /* CUDA - skip prologue */
    int cuda_producer_is_open64;
  };

/* Defines for the objfile flag word.  */

/* When an object file has its functions reordered (currently Irix-5.2
   shared libraries exhibit this behaviour), we will need an expensive
   algorithm to locate a partial symtab or symtab via an address.
   To avoid this penalty for normal object files, we use this flag,
   whose setting is determined upon symbol table read in.  */

#define OBJF_REORDERED	(1 << 0)	/* Functions are reordered */

/* Distinguish between an objfile for a shared library and a "vanilla"
   objfile.  (If not set, the objfile may still actually be a solib.
   This can happen if the user created the objfile by using the
   add-symbol-file command.  GDB doesn't in that situation actually
   check whether the file is a solib.  Rather, the target's
   implementation of the solib interface is responsible for setting
   this flag when noticing solibs used by an inferior.)  */

#define OBJF_SHARED     (1 << 1)	/* From a shared library */

/* User requested that this objfile be read in it's entirety.  */

#define OBJF_READNOW	(1 << 2)	/* Immediate full read */

/* This objfile was created because the user explicitly caused it
   (e.g., used the add-symbol-file command).  This bit offers a way
   for run_command to remove old objfile entries which are no longer
   valid (i.e., are associated with an old inferior), but to preserve
   ones that the user explicitly loaded via the add-symbol-file
   command.  */

#define OBJF_USERLOADED	(1 << 3)	/* User loaded */

/* Set if we have tried to read partial symtabs for this objfile.
   This is used to allow lazy reading of partial symtabs.  */

#define OBJF_PSYMTABS_READ (1 << 4)

/* Set if this is the main symbol file
   (as opposed to symbol file for dynamically loaded code).  */

#define OBJF_MAINLINE (1 << 5)

/* The object file that contains the runtime common minimal symbols
   for SunOS4.  Note that this objfile has no associated BFD.  */

extern struct objfile *rt_common_objfile;

/* Declarations for functions defined in objfiles.c */

extern struct objfile *allocate_objfile (bfd *, int);

extern struct gdbarch *get_objfile_arch (struct objfile *);

extern int entry_point_address_query (CORE_ADDR *entry_p);

extern CORE_ADDR entry_point_address (void);

extern void build_objfile_section_table (struct objfile *);

extern void terminate_minimal_symbol_table (struct objfile *objfile);

extern struct objfile *objfile_separate_debug_iterate (const struct objfile *,
                                                       const struct objfile *);

extern void put_objfile_before (struct objfile *, struct objfile *);

extern void objfile_to_front (struct objfile *);

extern void add_separate_debug_objfile (struct objfile *, struct objfile *);

extern void unlink_objfile (struct objfile *);

extern void free_objfile (struct objfile *);

extern void free_objfile_separate_debug (struct objfile *);

extern struct cleanup *make_cleanup_free_objfile (struct objfile *);

extern void free_all_objfiles (void);

extern void objfile_relocate (struct objfile *, struct section_offsets *);
extern void objfile_rebase (struct objfile *, CORE_ADDR);

extern int objfile_has_partial_symbols (struct objfile *objfile);

extern int objfile_has_full_symbols (struct objfile *objfile);

extern int objfile_has_symbols (struct objfile *objfile);

extern int have_partial_symbols (void);

extern int have_full_symbols (void);

extern void objfiles_changed (void);

/* This operation deletes all objfile entries that represent solibs that
   weren't explicitly loaded by the user, via e.g., the add-symbol-file
   command.  */

extern void objfile_purge_solibs (void);

/* Functions for dealing with the minimal symbol table, really a misc
   address<->symbol mapping for things we don't have debug symbols for.  */

extern int have_minimal_symbols (void);

extern struct obj_section *find_pc_section (CORE_ADDR pc);

extern int in_plt_section (CORE_ADDR, char *);

/* Keep a registry of per-objfile data-pointers required by other GDB
   modules.  */
DECLARE_REGISTRY(objfile);

extern void default_iterate_over_objfiles_in_search_order
  (struct gdbarch *gdbarch,
   iterate_over_objfiles_in_search_order_cb_ftype *cb,
   void *cb_data, struct objfile *current_objfile);


/* Traverse all object files in the current program space.
   ALL_OBJFILES_SAFE works even if you delete the objfile during the
   traversal.  */

/* Traverse all object files in program space SS.  */

#define ALL_PSPACE_OBJFILES(ss, obj)					\
  for ((obj) = ss->objfiles; (obj) != NULL; (obj) = (obj)->next)

#define ALL_PSPACE_OBJFILES_SAFE(ss, obj, nxt)		\
  for ((obj) = ss->objfiles;			\
       (obj) != NULL? ((nxt)=(obj)->next,1) :0;	\
       (obj) = (nxt))

#define ALL_OBJFILES(obj)			    \
  for ((obj) = current_program_space->objfiles; \
       (obj) != NULL;				    \
       (obj) = (obj)->next)

#define ALL_OBJFILES_SAFE(obj,nxt)			\
  for ((obj) = current_program_space->objfiles;	\
       (obj) != NULL? ((nxt)=(obj)->next,1) :0;	\
       (obj) = (nxt))

/* Traverse all symtabs in one objfile.  */

#define	ALL_OBJFILE_SYMTABS(objfile, s) \
    for ((s) = (objfile) -> symtabs; (s) != NULL; (s) = (s) -> next)

/* Traverse all primary symtabs in one objfile.  */

#define ALL_OBJFILE_PRIMARY_SYMTABS(objfile, s) \
  ALL_OBJFILE_SYMTABS ((objfile), (s)) \
    if ((s)->primary)

/* Traverse all minimal symbols in one objfile.  */

#define	ALL_OBJFILE_MSYMBOLS(objfile, m) \
    for ((m) = (objfile) -> msymbols; SYMBOL_LINKAGE_NAME(m) != NULL; (m)++)

/* Traverse all symtabs in all objfiles in the current symbol
   space.  */

#define	ALL_SYMTABS(objfile, s) \
  ALL_OBJFILES (objfile)	 \
    ALL_OBJFILE_SYMTABS (objfile, s)

#define ALL_PSPACE_SYMTABS(ss, objfile, s)		\
  ALL_PSPACE_OBJFILES (ss, objfile)			\
    ALL_OBJFILE_SYMTABS (objfile, s)

/* Traverse all symtabs in all objfiles in the current program space,
   skipping included files (which share a blockvector with their
   primary symtab).  */

#define ALL_PRIMARY_SYMTABS(objfile, s) \
  ALL_OBJFILES (objfile)		\
    ALL_OBJFILE_PRIMARY_SYMTABS (objfile, s)

#define ALL_PSPACE_PRIMARY_SYMTABS(pspace, objfile, s)	\
  ALL_PSPACE_OBJFILES (ss, objfile)			\
    ALL_OBJFILE_PRIMARY_SYMTABS (objfile, s)

/* Traverse all minimal symbols in all objfiles in the current symbol
   space.  */

#define	ALL_MSYMBOLS(objfile, m) \
  ALL_OBJFILES (objfile)	 \
    ALL_OBJFILE_MSYMBOLS (objfile, m)

#define ALL_OBJFILE_OSECTIONS(objfile, osect)	\
  for (osect = objfile->sections; osect < objfile->sections_end; osect++)

/* Traverse all obj_sections in all objfiles in the current program
   space.

   Note that this detects a "break" in the inner loop, and exits
   immediately from the outer loop as well, thus, client code doesn't
   need to know that this is implemented with a double for.  The extra
   hair is to make sure that a "break;" stops the outer loop iterating
   as well, and both OBJFILE and OSECT are left unmodified:

    - The outer loop learns about the inner loop's end condition, and
      stops iterating if it detects the inner loop didn't reach its
      end.  In other words, the outer loop keeps going only if the
      inner loop reached its end cleanly [(osect) ==
      (objfile)->sections_end].

    - OSECT is initialized in the outer loop initialization
      expressions, such as if the inner loop has reached its end, so
      the check mentioned above succeeds the first time.

    - The trick to not clearing OBJFILE on a "break;" is, in the outer
      loop's loop expression, advance OBJFILE, but iff the inner loop
      reached its end.  If not, there was a "break;", so leave OBJFILE
      as is; the outer loop's conditional will break immediately as
      well (as OSECT will be different from OBJFILE->sections_end).  */

#define ALL_OBJSECTIONS(objfile, osect)					\
  for ((objfile) = current_program_space->objfiles,			\
	 (objfile) != NULL ? ((osect) = (objfile)->sections_end) : 0;	\
       (objfile) != NULL						\
	 && (osect) == (objfile)->sections_end;				\
       ((osect) == (objfile)->sections_end				\
	? ((objfile) = (objfile)->next,					\
	   (objfile) != NULL ? (osect) = (objfile)->sections_end : 0)	\
	: 0))								\
    for ((osect) = (objfile)->sections;					\
	 (osect) < (objfile)->sections_end;				\
	 (osect)++)

#define SECT_OFF_DATA(objfile) \
     ((objfile->sect_index_data == -1) \
      ? (internal_error (__FILE__, __LINE__, \
			 _("sect_index_data not initialized")), -1)	\
      : objfile->sect_index_data)

#define SECT_OFF_RODATA(objfile) \
     ((objfile->sect_index_rodata == -1) \
      ? (internal_error (__FILE__, __LINE__, \
			 _("sect_index_rodata not initialized")), -1)	\
      : objfile->sect_index_rodata)

#define SECT_OFF_TEXT(objfile) \
     ((objfile->sect_index_text == -1) \
      ? (internal_error (__FILE__, __LINE__, \
			 _("sect_index_text not initialized")), -1)	\
      : objfile->sect_index_text)

/* Sometimes the .bss section is missing from the objfile, so we don't
   want to die here.  Let the users of SECT_OFF_BSS deal with an
   uninitialized section index.  */
#define SECT_OFF_BSS(objfile) (objfile)->sect_index_bss

/* Answer whether there is more than one object file loaded.  */

#define MULTI_OBJFILE_P() (object_files && object_files->next)

/* Reset the per-BFD storage area on OBJ.  */

void set_objfile_per_bfd (struct objfile *obj);

#endif /* !defined (OBJFILES_H) */
