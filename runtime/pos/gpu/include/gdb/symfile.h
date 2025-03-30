/* Definitions for reading symbol files into GDB.

   Copyright (C) 1990-2013 Free Software Foundation, Inc.

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

#if !defined (SYMFILE_H)
#define SYMFILE_H

/* This file requires that you first include "bfd.h".  */
#include "symtab.h"
#include "probe.h"

/* Opaque declarations.  */
struct target_section;
struct objfile;
struct obj_section;
struct obstack;
struct block;
struct probe;
struct value;
struct frame_info;
struct agent_expr;
struct axs_value;

/* Comparison function for symbol look ups.  */

typedef int (symbol_compare_ftype) (const char *string1,
				    const char *string2);

/* Partial symbols are stored in the psymbol_cache and pointers to
   them are kept in a dynamically grown array that is obtained from
   malloc and grown as necessary via realloc.  Each objfile typically
   has two of these, one for global symbols and one for static
   symbols.  Although this adds a level of indirection for storing or
   accessing the partial symbols, it allows us to throw away duplicate
   psymbols and set all pointers to the single saved instance.  */

struct psymbol_allocation_list
{

  /* Pointer to beginning of dynamically allocated array of pointers
     to partial symbols.  The array is dynamically expanded as
     necessary to accommodate more pointers.  */

  struct partial_symbol **list;

  /* Pointer to next available slot in which to store a pointer to a
     partial symbol.  */

  struct partial_symbol **next;

  /* Number of allocated pointer slots in current dynamic array (not
     the number of bytes of storage).  The "next" pointer will always
     point somewhere between list[0] and list[size], and when at
     list[size] the array will be expanded on the next attempt to
     store a pointer.  */

  int size;
};

/* Define an array of addresses to accommodate non-contiguous dynamic
   loading of modules.  This is for use when entering commands, so we
   can keep track of the section names until we read the file and can
   map them to bfd sections.  This structure is also used by solib.c
   to communicate the section addresses in shared objects to
   symbol_file_add ().  */

struct section_addr_info
{
  /* The number of sections for which address information is
     available.  */
  size_t num_sections;
  /* Sections whose names are file format dependent.  */
  struct other_sections
  {
    CORE_ADDR addr;
    char *name;

    /* SECTINDEX must be valid for associated BFD or set to -1.  */
    int sectindex;
  } other[1];
};


/* A table listing the load segments in a symfile, and which segment
   each BFD section belongs to.  */
struct symfile_segment_data
{
  /* How many segments are present in this file.  If there are
     two, the text segment is the first one and the data segment
     is the second one.  */
  int num_segments;

  /* If NUM_SEGMENTS is greater than zero, the original base address
     of each segment.  */
  CORE_ADDR *segment_bases;

  /* If NUM_SEGMENTS is greater than zero, the memory size of each
     segment.  */
  CORE_ADDR *segment_sizes;

  /* If NUM_SEGMENTS is greater than zero, this is an array of entries
     recording which segment contains each BFD section.
     SEGMENT_INFO[I] is S+1 if the I'th BFD section belongs to segment
     S, or zero if it is not in any segment.  */
  int *segment_info;
};

/* Callback for quick_symbol_functions->map_symbol_filenames.  */

typedef void (symbol_filename_ftype) (const char *filename,
				      const char *fullname, void *data);

/* The "quick" symbol functions exist so that symbol readers can
   avoiding an initial read of all the symbols.  For example, symbol
   readers might choose to use the "partial symbol table" utilities,
   which is one implementation of the quick symbol functions.

   The quick symbol functions are generally opaque: the underlying
   representation is hidden from the caller.

   In general, these functions should only look at whatever special
   index the symbol reader creates -- looking through the symbol
   tables themselves is handled by generic code.  If a function is
   defined as returning a "symbol table", this means that the function
   should only return a newly-created symbol table; it should not
   examine pre-existing ones.

   The exact list of functions here was determined in an ad hoc way
   based on gdb's history.  */

struct quick_symbol_functions
{
  /* Return true if this objfile has any "partial" symbols
     available.  */
  int (*has_symbols) (struct objfile *objfile);

  /* Return the symbol table for the "last" file appearing in
     OBJFILE.  */
  struct symtab *(*find_last_source_symtab) (struct objfile *objfile);

  /* Forget all cached full file names for OBJFILE.  */
  void (*forget_cached_source_info) (struct objfile *objfile);

  /* Expand and iterate over each "partial" symbol table in OBJFILE
     where the source file is named NAME.

     If NAME is not absolute, a match after a '/' in the symbol table's
     file name will also work, REAL_PATH is NULL then.  If NAME is
     absolute then REAL_PATH is non-NULL absolute file name as resolved
     via gdb_realpath from NAME.

     If a match is found, the "partial" symbol table is expanded.
     Then, this calls iterate_over_some_symtabs (or equivalent) over
     all newly-created symbol tables, passing CALLBACK and DATA to it.
     The result of this call is returned.  */
  int (*map_symtabs_matching_filename) (struct objfile *objfile,
					const char *name,
					const char *real_path,
					int (*callback) (struct symtab *,
							 void *),
					void *data);

  /* Check to see if the symbol is defined in a "partial" symbol table
     of OBJFILE.  KIND should be either GLOBAL_BLOCK or STATIC_BLOCK,
     depending on whether we want to search global symbols or static
     symbols.  NAME is the name of the symbol to look for.  DOMAIN
     indicates what sort of symbol to search for.

     Returns the newly-expanded symbol table in which the symbol is
     defined, or NULL if no such symbol table exists.  If OBJFILE
     contains !TYPE_OPAQUE symbol prefer its symtab.  If it contains
     only TYPE_OPAQUE symbol(s), return at least that symtab.  */
  struct symtab *(*lookup_symbol) (struct objfile *objfile,
				   int kind, const char *name,
				   domain_enum domain);

  /* Print statistics about any indices loaded for OBJFILE.  The
     statistics should be printed to gdb_stdout.  This is used for
     "maint print statistics".  */
  void (*print_stats) (struct objfile *objfile);

  /* Dump any indices loaded for OBJFILE.  The dump should go to
     gdb_stdout.  This is used for "maint print objfiles".  */
  void (*dump) (struct objfile *objfile);

  /* This is called by objfile_relocate to relocate any indices loaded
     for OBJFILE.  */
  void (*relocate) (struct objfile *objfile,
		    struct section_offsets *new_offsets,
		    struct section_offsets *delta);

  /* Find all the symbols in OBJFILE named FUNC_NAME, and ensure that
     the corresponding symbol tables are loaded.  */
  void (*expand_symtabs_for_function) (struct objfile *objfile,
				       const char *func_name);

  /* Read all symbol tables associated with OBJFILE.  */
  void (*expand_all_symtabs) (struct objfile *objfile);

  /* Read all symbol tables associated with OBJFILE which have
     symtab_to_fullname equal to FULLNAME.
     This is for the purposes of examining code only, e.g., expand_line_sal.
     The routine may ignore debug info that is known to not be useful with
     code, e.g., DW_TAG_type_unit for dwarf debug info.  */
  void (*expand_symtabs_with_fullname) (struct objfile *objfile,
					const char *fullname);

  /* Return the file name of the file holding the global symbol in OBJFILE
     named NAME.  If no such symbol exists in OBJFILE, return NULL.
     Only file extension of returned filename is recognized.  */
  const char *(*find_symbol_file) (struct objfile *objfile, const char *name);

  /* Find global or static symbols in all tables that are in NAMESPACE 
     and for which MATCH (symbol name, NAME) == 0, passing each to 
     CALLBACK, reading in partial symbol tables as needed.  Look
     through global symbols if GLOBAL and otherwise static symbols.
     Passes NAME, NAMESPACE, and DATA to CALLBACK with each symbol
     found.  After each block is processed, passes NULL to CALLBACK.
     MATCH must be weaker than strcmp_iw_ordered in the sense that
     strcmp_iw_ordered(x,y) == 0 --> MATCH(x,y) == 0.  ORDERED_COMPARE,
     if non-null, must be an ordering relation compatible with
     strcmp_iw_ordered in the sense that
            strcmp_iw_ordered(x,y) == 0 --> ORDERED_COMPARE(x,y) == 0
     and 
            strcmp_iw_ordered(x,y) <= 0 --> ORDERED_COMPARE(x,y) <= 0
     (allowing strcmp_iw_ordered(x,y) < 0 while ORDERED_COMPARE(x, y) == 0).
     CALLBACK returns 0 to indicate that the scan should continue, or
     non-zero to indicate that the scan should be terminated.  */

  void (*map_matching_symbols) (const char *name, domain_enum namespace,
				struct objfile *, int global,
				int (*callback) (struct block *,
						 struct symbol *, void *),
				void *data,
				symbol_compare_ftype *match,
				symbol_compare_ftype *ordered_compare);

  /* Expand all symbol tables in OBJFILE matching some criteria.

     FILE_MATCHER is called for each file in OBJFILE.  The file name
     and the DATA argument are passed to it.  If it returns zero, this
     file is skipped.  If FILE_MATCHER is NULL such file is not skipped.
     If BASENAMES is non-zero the function should consider only base name of
     DATA (passed file name is already only the lbasename part).

     Otherwise, if KIND does not match this symbol is skipped.

     If even KIND matches, then NAME_MATCHER is called for each symbol
     defined in the file.  The symbol "search" name and DATA are passed
     to NAME_MATCHER.

     If NAME_MATCHER returns zero, then this symbol is skipped.

     Otherwise, this symbol's symbol table is expanded.

     DATA is user data that is passed unmodified to the callback
     functions.  */
  void (*expand_symtabs_matching)
    (struct objfile *objfile,
     int (*file_matcher) (const char *, void *, int basenames),
     int (*name_matcher) (const char *, void *),
     enum search_domain kind,
     void *data);

  /* Return the symbol table from OBJFILE that contains PC and
     SECTION.  Return NULL if there is no such symbol table.  This
     should return the symbol table that contains a symbol whose
     address exactly matches PC, or, if there is no exact match, the
     symbol table that contains a symbol whose address is closest to
     PC.  */
  struct symtab *(*find_pc_sect_symtab) (struct objfile *objfile,
					 struct minimal_symbol *msymbol,
					 CORE_ADDR pc,
					 struct obj_section *section,
					 int warn_if_readin);

  /* Call a callback for every file defined in OBJFILE whose symtab is
     not already read in.  FUN is the callback.  It is passed the file's
     FILENAME, the file's FULLNAME (if need_fullname is non-zero), and
     the DATA passed to this function.  */
  void (*map_symbol_filenames) (struct objfile *objfile,
				symbol_filename_ftype *fun, void *data,
				int need_fullname);
};

/* Structure of functions used for probe support.  If one of these functions
   is provided, all must be.  */

struct sym_probe_fns
{
  /* If non-NULL, return an array of probe objects.

     The returned value does not have to be freed and it has lifetime of the
     OBJFILE.  */
  VEC (probe_p) *(*sym_get_probes) (struct objfile *);

  /* Return the number of arguments available to PROBE.  PROBE will
     have come from a call to this objfile's sym_get_probes method.
     If you provide an implementation of sym_get_probes, you must
     implement this method as well.  */
  unsigned (*sym_get_probe_argument_count) (struct probe *probe);

  /* Evaluate the Nth argument available to PROBE.  PROBE will have
     come from a call to this objfile's sym_get_probes method.  N will
     be between 0 and the number of arguments available to this probe.
     FRAME is the frame in which the evaluation is done; the frame's
     PC will match the address of the probe.  If you provide an
     implementation of sym_get_probes, you must implement this method
     as well.  */
  struct value *(*sym_evaluate_probe_argument) (struct probe *probe,
						unsigned n);

  /* Compile the Nth probe argument to an agent expression.  PROBE
     will have come from a call to this objfile's sym_get_probes
     method.  N will be between 0 and the number of arguments
     available to this probe.  EXPR and VALUE are the agent expression
     that is being updated.  */
  void (*sym_compile_to_ax) (struct probe *probe,
			     struct agent_expr *expr,
			     struct axs_value *value,
			     unsigned n);

  /* Relocate the probe section of OBJFILE.  */
  void (*sym_relocate_probe) (struct objfile *objfile,
			      struct section_offsets *new_offsets,
			      struct section_offsets *delta);
};

/* Structure to keep track of symbol reading functions for various
   object file types.  */

struct sym_fns
{

  /* BFD flavour that we handle, or (as a special kludge, see
     xcoffread.c, (enum bfd_flavour)-1 for xcoff).  */

  enum bfd_flavour sym_flavour;

  /* Initializes anything that is global to the entire symbol table.
     It is called during symbol_file_add, when we begin debugging an
     entirely new program.  */

  void (*sym_new_init) (struct objfile *);

  /* Reads any initial information from a symbol file, and initializes
     the struct sym_fns SF in preparation for sym_read().  It is
     called every time we read a symbol file for any reason.  */

  void (*sym_init) (struct objfile *);

  /* sym_read (objfile, symfile_flags) Reads a symbol file into a psymtab
     (or possibly a symtab).  OBJFILE is the objfile struct for the
     file we are reading.  SYMFILE_FLAGS are the flags passed to
     symbol_file_add & co.  */

  void (*sym_read) (struct objfile *, int);

  /* Read the partial symbols for an objfile.  This may be NULL, in which case
     gdb has to check other ways if this objfile has any symbols.  This may
     only be non-NULL if the objfile actually does have debuginfo available.
     */

  void (*sym_read_psymbols) (struct objfile *);

  /* Called when we are finished with an objfile.  Should do all
     cleanup that is specific to the object file format for the
     particular objfile.  */

  void (*sym_finish) (struct objfile *);

  /* This function produces a file-dependent section_offsets
     structure, allocated in the objfile's storage, and based on the
     parameter.  The parameter is currently a CORE_ADDR (FIXME!) for
     backward compatibility with the higher levels of GDB.  It should
     probably be changed to a string, where NULL means the default,
     and others are parsed in a file dependent way.  */

  void (*sym_offsets) (struct objfile *, struct section_addr_info *);

  /* This function produces a format-independent description of
     the segments of ABFD.  Each segment is a unit of the file
     which may be relocated independently.  */

  struct symfile_segment_data *(*sym_segments) (bfd *abfd);

  /* This function should read the linetable from the objfile when
     the line table cannot be read while processing the debugging
     information.  */

  void (*sym_read_linetable) (void);

  /* Relocate the contents of a debug section SECTP.  The
     contents are stored in BUF if it is non-NULL, or returned in a
     malloc'd buffer otherwise.  */

  bfd_byte *(*sym_relocate) (struct objfile *, asection *sectp, bfd_byte *buf);

  /* If non-NULL, this objfile has probe support, and all the probe
     functions referred to here will be non-NULL.  */
  const struct sym_probe_fns *sym_probe_fns;

  /* The "quick" (aka partial) symbol functions for this symbol
     reader.  */
  const struct quick_symbol_functions *qf;
};

extern struct section_addr_info *
  build_section_addr_info_from_objfile (const struct objfile *objfile);

extern void relative_addr_info_to_section_offsets
  (struct section_offsets *section_offsets, int num_sections,
   struct section_addr_info *addrs);

extern void addr_info_make_relative (struct section_addr_info *addrs,
				     bfd *abfd);

/* The default version of sym_fns.sym_offsets for readers that don't
   do anything special.  */

extern void default_symfile_offsets (struct objfile *objfile,
				     struct section_addr_info *);

/* The default version of sym_fns.sym_segments for readers that don't
   do anything special.  */

extern struct symfile_segment_data *default_symfile_segments (bfd *abfd);

/* The default version of sym_fns.sym_relocate for readers that don't
   do anything special.  */

extern bfd_byte *default_symfile_relocate (struct objfile *objfile,
                                           asection *sectp, bfd_byte *buf);

extern struct symtab *allocate_symtab (const char *, struct objfile *)
  ATTRIBUTE_NONNULL (1);

extern void add_symtab_fns (const struct sym_fns *);

/* This enum encodes bit-flags passed as ADD_FLAGS parameter to
   syms_from_objfile, symbol_file_add, etc.  */

enum symfile_add_flags
  {
    /* Be chatty about what you are doing.  */
    SYMFILE_VERBOSE = 1 << 1,

    /* This is the main symbol file (as opposed to symbol file for dynamically
       loaded code).  */
    SYMFILE_MAINLINE = 1 << 2,

    /* Do not call breakpoint_re_set when adding this symbol file.  */
    SYMFILE_DEFER_BP_RESET = 1 << 3,

    /* Do not immediately read symbols for this file.  By default,
       symbols are read when the objfile is created.  */
    SYMFILE_NO_READ = 1 << 4
  };

extern void syms_from_objfile (struct objfile *,
			       struct section_addr_info *,
			       struct section_offsets *, int, int);

extern void new_symfile_objfile (struct objfile *, int);

extern struct objfile *symbol_file_add (char *, int,
					struct section_addr_info *, int);

extern struct objfile *symbol_file_add_from_bfd (bfd *, int,
                                                 struct section_addr_info *,
                                                 int, struct objfile *parent);

extern void symbol_file_add_separate (bfd *, int, struct objfile *);

extern char *find_separate_debug_file_by_debuglink (struct objfile *);

/* Create a new section_addr_info, with room for NUM_SECTIONS.  */

extern struct section_addr_info *alloc_section_addr_info (size_t
							  num_sections);

/* Build (allocate and populate) a section_addr_info struct from an
   existing section table.  */

extern struct section_addr_info
  *build_section_addr_info_from_section_table (const struct target_section
					       *start,
					       const struct target_section
					       *end);

/* Free all memory allocated by
   build_section_addr_info_from_section_table.  */

extern void free_section_addr_info (struct section_addr_info *);


			/*   Variables   */

/* If non-zero, shared library symbols will be added automatically
   when the inferior is created, new libraries are loaded, or when
   attaching to the inferior.  This is almost always what users will
   want to have happen; but for very large programs, the startup time
   will be excessive, and so if this is a problem, the user can clear
   this flag and then add the shared library symbols as needed.  Note
   that there is a potential for confusion, since if the shared
   library symbols are not loaded, commands like "info fun" will *not*
   report all the functions that are actually present.  */

extern int auto_solib_add;

/* From symfile.c */

extern void set_initial_language (void);

extern void find_lowest_section (bfd *, asection *, void *);

extern bfd *symfile_bfd_open (char *);

extern bfd *gdb_bfd_open_maybe_remote (const char *);

extern int get_section_index (struct objfile *, char *);

/* Utility functions for overlay sections: */
extern enum overlay_debugging_state
{
  ovly_off,
  ovly_on,
  ovly_auto
} overlay_debugging;
extern int overlay_cache_invalid;

/* Return the "mapped" overlay section containing the PC.  */
extern struct obj_section *find_pc_mapped_section (CORE_ADDR);

/* Return any overlay section containing the PC (even in its LMA
   region).  */
extern struct obj_section *find_pc_overlay (CORE_ADDR);

/* Return true if the section is an overlay.  */
extern int section_is_overlay (struct obj_section *);

/* Return true if the overlay section is currently "mapped".  */
extern int section_is_mapped (struct obj_section *);

/* Return true if pc belongs to section's VMA.  */
extern CORE_ADDR pc_in_mapped_range (CORE_ADDR, struct obj_section *);

/* Return true if pc belongs to section's LMA.  */
extern CORE_ADDR pc_in_unmapped_range (CORE_ADDR, struct obj_section *);

/* Map an address from a section's LMA to its VMA.  */
extern CORE_ADDR overlay_mapped_address (CORE_ADDR, struct obj_section *);

/* Map an address from a section's VMA to its LMA.  */
extern CORE_ADDR overlay_unmapped_address (CORE_ADDR, struct obj_section *);

/* Convert an address in an overlay section (force into VMA range).  */
extern CORE_ADDR symbol_overlayed_address (CORE_ADDR, struct obj_section *);

/* Load symbols from a file.  */
extern void symbol_file_add_main (char *args, int from_tty);

/* Clear GDB symbol tables.  */
extern void symbol_file_clear (int from_tty);

/* Default overlay update function.  */
extern void simple_overlay_update (struct obj_section *);

extern bfd_byte *symfile_relocate_debug_section (struct objfile *, asection *,
						 bfd_byte *);

extern int symfile_map_offsets_to_segments (bfd *,
					    struct symfile_segment_data *,
					    struct section_offsets *,
					    int, const CORE_ADDR *);
struct symfile_segment_data *get_symfile_segment_data (bfd *abfd);
void free_symfile_segment_data (struct symfile_segment_data *data);

extern struct cleanup *increment_reading_symtab (void);

/* From dwarf2read.c */

/* Names for a dwarf2 debugging section.  The field NORMAL is the normal
   section name (usually from the DWARF standard), while the field COMPRESSED
   is the name of compressed sections.  If your object file format doesn't
   support compressed sections, the field COMPRESSED can be NULL.  Likewise,
   the debugging section is not supported, the field NORMAL can be NULL too.
   It doesn't make sense to have a NULL NORMAL field but a non-NULL COMPRESSED
   field.  */

struct dwarf2_section_names {
  const char *normal;
  const char *compressed;
};

/* List of names for dward2 debugging sections.  Also most object file formats
   use the standardized (ie ELF) names, some (eg XCOFF) have customized names
   due to restrictions.
   The table for the standard names is defined in dwarf2read.c.  Please
   update all instances of dwarf2_debug_sections if you add a field to this
   structure.  It is always safe to use { NULL, NULL } in this case.  */

struct dwarf2_debug_sections {
  struct dwarf2_section_names info;
  struct dwarf2_section_names abbrev;
  struct dwarf2_section_names line;
  struct dwarf2_section_names loc;
  struct dwarf2_section_names macinfo;
  struct dwarf2_section_names macro;
  struct dwarf2_section_names str;
  struct dwarf2_section_names ranges;
  struct dwarf2_section_names types;
  struct dwarf2_section_names addr;
  struct dwarf2_section_names frame;
  struct dwarf2_section_names eh_frame;
  struct dwarf2_section_names gdb_index;
  /* This field has no meaning, but exists solely to catch changes to
     this structure which are not reflected in some instance.  */
  int sentinel;
};

extern int dwarf2_has_info (struct objfile *,
                            const struct dwarf2_debug_sections *);

/* Dwarf2 sections that can be accessed by dwarf2_get_section_info.  */
enum dwarf2_section_enum {
  DWARF2_DEBUG_FRAME,
  DWARF2_EH_FRAME
};

extern void dwarf2_get_section_info (struct objfile *,
                                     enum dwarf2_section_enum,
				     asection **, gdb_byte **,
				     bfd_size_type *);

extern int dwarf2_initialize_objfile (struct objfile *);
extern void dwarf2_build_psymtabs (struct objfile *);
extern void dwarf2_build_frame_info (struct objfile *);

void dwarf2_free_objfile (struct objfile *);

/* From mdebugread.c */

extern void mdebug_build_psymtabs (struct objfile *,
				   const struct ecoff_debug_swap *,
				   struct ecoff_debug_info *);

extern void elfmdebug_build_psymtabs (struct objfile *,
				      const struct ecoff_debug_swap *,
				      asection *);

/* From minidebug.c.  */

extern bfd *find_separate_debug_file_in_section (struct objfile *);

#endif /* !defined(SYMFILE_H) */
