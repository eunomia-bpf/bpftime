/* *INDENT-OFF* */ /* ATTRIBUTE_PRINTF confuses indent, avoid running it
		      for now.  */
/* I/O, string, cleanup, and other random utilities for GDB.
   Copyright (C) 1986-2013 Free Software Foundation, Inc.

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

#ifndef UTILS_H
#define UTILS_H

#include "cleanups.h"
#include "exceptions.h"

extern void initialize_utils (void);

/* String utilities.  */

extern int sevenbit_strings;

extern char *savestring (const char *, size_t);

extern int strcmp_iw (const char *, const char *);

extern int strcmp_iw_ordered (const char *, const char *);

extern int streq (const char *, const char *);

extern int subset_compare (char *, char *);

ULONGEST strtoulst (const char *num, const char **trailer, int base);

int compare_positive_ints (const void *ap, const void *bp);
int compare_strings (const void *ap, const void *bp);

/* This is defined in *-hdep.c, e.g., posix-hdep.c.  */
extern char *safe_strerror (int);

/* A wrapper for bfd_errmsg to produce a more helpful error message
   in the case of bfd_error_file_ambiguously recognized.
   MATCHING, if non-NULL, is the corresponding argument to
   bfd_check_format_matches, and will be freed.  */

extern const char *gdb_bfd_errmsg (bfd_error_type error_tag, char **matching);

/* Parsing utilites.  */

extern int parse_pid_to_attach (char *args);

extern int parse_escape (struct gdbarch *, char **);

char **gdb_buildargv (const char *);

/* Cleanup utilities.  */

extern struct cleanup *make_cleanup_freeargv (char **);

struct dyn_string;
extern struct cleanup *make_cleanup_dyn_string_delete (struct dyn_string *);

struct ui_file;
extern struct cleanup *make_cleanup_ui_file_delete (struct ui_file *);

struct ui_out;
extern struct cleanup *
  make_cleanup_ui_out_redirect_pop (struct ui_out *uiout);

struct section_addr_info;
extern struct cleanup *(make_cleanup_free_section_addr_info 
                        (struct section_addr_info *));

extern struct cleanup *make_cleanup_close (int fd);

extern struct cleanup *make_cleanup_fclose (FILE *file);

extern struct cleanup *make_cleanup_bfd_unref (bfd *abfd);

struct obstack;
extern struct cleanup *make_cleanup_obstack_free (struct obstack *obstack);

extern struct cleanup *make_cleanup_restore_integer (int *variable);
extern struct cleanup *make_cleanup_restore_uinteger (unsigned int *variable);

struct target_ops;
extern struct cleanup *make_cleanup_unpush_target (struct target_ops *ops);

extern struct cleanup *
  make_cleanup_restore_ui_file (struct ui_file **variable);

extern struct cleanup *make_cleanup_value_free_to_mark (struct value *);
extern struct cleanup *make_cleanup_value_free (struct value *);

struct so_list;
extern struct cleanup *make_cleanup_free_so (struct so_list *so);

extern struct cleanup *make_cleanup_restore_current_language (void);

extern struct cleanup *make_cleanup_htab_delete (htab_t htab);

extern void free_current_contents (void *);

extern struct cleanup *make_command_stats_cleanup (int);

extern void init_page_info (void);

extern struct cleanup *make_cleanup_restore_page_info (void);
extern struct cleanup *
  set_batch_flag_and_make_cleanup_restore_page_info (void);

extern struct cleanup *make_bpstat_clear_actions_cleanup (void);

/* Path utilities.  */

extern char *gdb_realpath (const char *);

extern int gdb_filename_fnmatch (const char *pattern, const char *string,
				 int flags);

extern void substitute_path_component (char **stringp, const char *from,
				       const char *to);

char *ldirname (const char *filename);

/* GDB output, ui_file utilities.  */

struct ui_file;

extern void set_display_time (int);

extern void set_display_space (int);

extern int query (const char *, ...) ATTRIBUTE_PRINTF (1, 2);
extern int nquery (const char *, ...) ATTRIBUTE_PRINTF (1, 2);
extern int yquery (const char *, ...) ATTRIBUTE_PRINTF (1, 2);

extern void begin_line (void);

extern void wrap_here (char *);

extern void reinitialize_more_filter (void);

/* Global ui_file streams.  These are all defined in main.c.  */
/* Normal results */
extern struct ui_file *gdb_stdout;
/* Input stream */
extern struct ui_file *gdb_stdin;
/* Serious error notifications */
extern struct ui_file *gdb_stderr;
/* Log/debug/trace messages that should bypass normal stdout/stderr
   filtering.  For moment, always call this stream using
   *_unfiltered.  In the very near future that restriction shall be
   removed - either call shall be unfiltered.  (cagney 1999-06-13).  */
extern struct ui_file *gdb_stdlog;
/* Target output that should bypass normal stdout/stderr filtering.
   For moment, always call this stream using *_unfiltered.  In the
   very near future that restriction shall be removed - either call
   shall be unfiltered.  (cagney 1999-07-02).  */
extern struct ui_file *gdb_stdtarg;
extern struct ui_file *gdb_stdtargerr;
extern struct ui_file *gdb_stdtargin;

/* More generic printf like operations.  Filtered versions may return
   non-locally on error.  */

extern void fputs_filtered (const char *, struct ui_file *);

extern void fputs_unfiltered (const char *, struct ui_file *);

extern int fputc_filtered (int c, struct ui_file *);

extern int fputc_unfiltered (int c, struct ui_file *);

extern int putchar_filtered (int c);

extern int putchar_unfiltered (int c);

extern void puts_filtered (const char *);

extern void puts_unfiltered (const char *);

extern void puts_filtered_tabular (char *string, int width, int right);

extern void puts_debug (char *prefix, char *string, char *suffix);

extern void vprintf_filtered (const char *, va_list) ATTRIBUTE_PRINTF (1, 0);

extern void vfprintf_filtered (struct ui_file *, const char *, va_list)
  ATTRIBUTE_PRINTF (2, 0);

extern void fprintf_filtered (struct ui_file *, const char *, ...)
  ATTRIBUTE_PRINTF (2, 3);

extern void fprintfi_filtered (int, struct ui_file *, const char *, ...)
  ATTRIBUTE_PRINTF (3, 4);

extern void printf_filtered (const char *, ...) ATTRIBUTE_PRINTF (1, 2);

extern void printfi_filtered (int, const char *, ...) ATTRIBUTE_PRINTF (2, 3);

extern void vprintf_unfiltered (const char *, va_list) ATTRIBUTE_PRINTF (1, 0);

extern void vfprintf_unfiltered (struct ui_file *, const char *, va_list)
  ATTRIBUTE_PRINTF (2, 0);

extern void fprintf_unfiltered (struct ui_file *, const char *, ...)
  ATTRIBUTE_PRINTF (2, 3);

extern void printf_unfiltered (const char *, ...) ATTRIBUTE_PRINTF (1, 2);

extern void print_spaces (int, struct ui_file *);

extern void print_spaces_filtered (int, struct ui_file *);

extern char *n_spaces (int);

extern void fputstr_filtered (const char *str, int quotr,
			      struct ui_file * stream);

extern void fputstr_unfiltered (const char *str, int quotr,
				struct ui_file * stream);

extern void fputstrn_filtered (const char *str, int n, int quotr,
			       struct ui_file * stream);

extern void fputstrn_unfiltered (const char *str, int n, int quotr,
				 struct ui_file * stream);

/* Display the host ADDR on STREAM formatted as ``0x%x''.  */
extern void gdb_print_host_address (const void *addr, struct ui_file *stream);

extern const char *host_address_to_string (const void *addr);

/* Convert CORE_ADDR to string in platform-specific manner.
   This is usually formatted similar to 0x%lx.  */
extern const char *paddress (struct gdbarch *gdbarch, CORE_ADDR addr);

/* Return a string representation in hexadecimal notation of ADDRESS,
   which is suitable for printing.  */

extern const char *print_core_address (struct gdbarch *gdbarch,
				       CORE_ADDR address);

/* Callback hash_f and eq_f for htab_create_alloc or htab_create_alloc_ex.  */
extern hashval_t core_addr_hash (const void *ap);
extern int core_addr_eq (const void *ap, const void *bp);

/* %d for LONGEST */
extern char *plongest (LONGEST l);
/* %u for ULONGEST */
extern char *pulongest (ULONGEST l);

extern char *phex (ULONGEST l, int sizeof_l);
extern char *phex_nz (ULONGEST l, int sizeof_l);
extern char *int_string (LONGEST, int, int, int, int);

/* Convert a CORE_ADDR into a HEX string with leading zeros.
   The output from core_addr_to_string() can be passed direct to
   string_to_core_addr().  */
extern const char *core_addr_to_string (const CORE_ADDR addr);
extern const char *core_addr_to_string_nz (const CORE_ADDR addr);
extern CORE_ADDR string_to_core_addr (const char *my_string);

/* Return a string that contains a number formatted as a hex
   string.  */
extern char *hex_string (ULONGEST);
extern char *hex_string_custom (LONGEST, int);

extern void fprintf_symbol_filtered (struct ui_file *, const char *,
				     enum language, int);

extern void throw_perror_with_name (enum errors errcode, const char *string)
  ATTRIBUTE_NORETURN;
extern void perror_with_name (const char *) ATTRIBUTE_NORETURN;

extern void print_sys_errmsg (const char *, int);

/* Warnings and error messages.  */

extern void (*deprecated_error_begin_hook) (void);

/* Message to be printed before the error message, when an error occurs.  */

extern char *error_pre_print;

/* Message to be printed before the error message, when an error occurs.  */

extern char *quit_pre_print;

/* Message to be printed before the warning message, when a warning occurs.  */

extern char *warning_pre_print;

extern void verror (const char *fmt, va_list ap)
     ATTRIBUTE_NORETURN ATTRIBUTE_PRINTF (1, 0);

extern void error (const char *fmt, ...)
     ATTRIBUTE_NORETURN ATTRIBUTE_PRINTF (1, 2);

extern void error_stream (struct ui_file *) ATTRIBUTE_NORETURN;

extern void vfatal (const char *fmt, va_list ap)
     ATTRIBUTE_NORETURN ATTRIBUTE_PRINTF (1, 0);

extern void fatal (const char *fmt, ...)
     ATTRIBUTE_NORETURN ATTRIBUTE_PRINTF (1, 2);

extern void internal_verror (const char *file, int line, const char *,
			     va_list ap)
     ATTRIBUTE_NORETURN ATTRIBUTE_PRINTF (3, 0);

extern void internal_vwarning (const char *file, int line,
			       const char *, va_list ap)
     ATTRIBUTE_PRINTF (3, 0);

extern void internal_warning (const char *file, int line,
			      const char *, ...) ATTRIBUTE_PRINTF (3, 4);

extern void warning (const char *, ...) ATTRIBUTE_PRINTF (1, 2);

extern void vwarning (const char *, va_list args) ATTRIBUTE_PRINTF (1, 0);

/* Misc. utilities.  */

/* Allocation and deallocation functions for the libiberty hash table
   which use obstacks.  */
void *hashtab_obstack_allocate (void *data, size_t size, size_t count);
void dummy_obstack_deallocate (void *object, void *data);

#ifdef HAVE_WAITPID
extern pid_t wait_to_die_with_timeout (pid_t pid, int *status, int timeout);
#endif

extern int producer_is_gcc_ge_4 (const char *producer);

extern int myread (int, char *, int);

/* Ensure that V is aligned to an N byte boundary (B's assumed to be a
   power of 2).  Round up/down when necessary.  Examples of correct
   use include:

   addr = align_up (addr, 8); -- VALUE needs 8 byte alignment
   write_memory (addr, value, len);
   addr += len;

   and:

   sp = align_down (sp - len, 16); -- Keep SP 16 byte aligned
   write_memory (sp, value, len);

   Note that uses such as:

   write_memory (addr, value, len);
   addr += align_up (len, 8);

   and:

   sp -= align_up (len, 8);
   write_memory (sp, value, len);

   are typically not correct as they don't ensure that the address (SP
   or ADDR) is correctly aligned (relying on previous alignment to
   keep things right).  This is also why the methods are called
   "align_..." instead of "round_..." as the latter reads better with
   this incorrect coding style.  */

extern ULONGEST align_up (ULONGEST v, int n);
extern ULONGEST align_down (ULONGEST v, int n);

#endif /* UTILS_H */
