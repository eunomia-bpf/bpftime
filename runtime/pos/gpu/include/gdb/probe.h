/* Generic SDT probe support for GDB.

   Copyright (C) 2012-2013 Free Software Foundation, Inc.

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

#if !defined (PROBE_H)
#define PROBE_H 1

#include "gdb_vecs.h"

/* Definition of a vector of probes.  */

typedef struct probe *probe_p;
DEF_VEC_P (probe_p);

struct linespec_result;

/* Structure useful for passing the header names in the method
   `gen_ui_out_table_header'.  */

struct info_probe_column
  {
    /* The internal name of the field.  This string cannot be capitalized nor
       localized, e.g., "extra_field".  */

    const char *field_name;

    /* The field name to be printed in the `info probes' command.  This
       string can be capitalized and localized, e.g., _("Extra Field").  */
    const char *print_name;
  };

typedef struct info_probe_column info_probe_column_s;
DEF_VEC_O (info_probe_column_s);

/* Operations associated with a probe.  */

struct probe_ops
  {
    /* Method responsible for verifying if LINESPECP is a valid linespec for
       a probe breakpoint.  It should return 1 if it is, or zero if it is not.
       It also should update LINESPECP in order to discard the breakpoint
       option associated with this linespec.  For example, if the option is
       `-probe', and the LINESPECP is `-probe abc', the function should
       return 1 and set LINESPECP to `abc'.  */

    int (*is_linespec) (const char **linespecp);

    /* Function that should fill PROBES with known probes from OBJFILE.  */

    void (*get_probes) (VEC (probe_p) **probes, struct objfile *objfile);

    /* Function used to relocate addresses from PROBE according to some DELTA
       provided.  */

    void (*relocate) (struct probe *probe, CORE_ADDR delta);

    /* Return the number of arguments of PROBE.  */

    unsigned (*get_probe_argument_count) (struct probe *probe);

    /* Evaluate the Nth argument from the PROBE, returning a value
       corresponding to it.  The argument number is represented N.  */

    struct value *(*evaluate_probe_argument) (struct probe *probe,
					      unsigned n);

    /* Compile the Nth argument of the PROBE to an agent expression.
       The argument number is represented by N.  */

    void (*compile_to_ax) (struct probe *probe, struct agent_expr *aexpr,
			   struct axs_value *axs_value, unsigned n);

    /* Set the semaphore associated with the PROBE.  This function only makes
       sense if the probe has a concept of semaphore associated to a
       probe.  */

    void (*set_semaphore) (struct probe *probe, struct gdbarch *gdbarch);

    /* Clear the semaphore associated with the PROBE.  This function only
       makes sense if the probe has a concept of semaphore associated to
       a probe.  */

    void (*clear_semaphore) (struct probe *probe, struct gdbarch *gdbarch);

    /* Function called to destroy PROBE's specific data.  This function
       shall not free PROBE itself.  */

    void (*destroy) (struct probe *probe);

    /* Function responsible for providing the extra fields that will be
       printed in the `info probes' command.  It should fill HEADS
       with whatever extra fields it needs.  If the backend doesn't need
       to print extra fields, it can set this method to NULL.  */

    void (*gen_info_probes_table_header) (VEC (info_probe_column_s) **heads);

    /* Function that will fill VALUES with the values of the extra fields
       to be printed for PROBE.  If the backend implements the
       `gen_ui_out_table_header' method, then it should implement
       this method as well.  The backend should also guarantee that the
       order and the number of values in the vector is exactly the same
       as the order of the extra fields provided in the method
       `gen_ui_out_table_header'.  If a certain field is to be skipped
       when printing the information, you can push a NULL value in that
       position in the vector.  */

    void (*gen_info_probes_table_values) (struct probe *probe,
					  VEC (const_char_ptr) **values);
  };

/* Definition of a vector of probe_ops.  */

typedef const struct probe_ops *probe_ops_cp;
DEF_VEC_P (probe_ops_cp);
extern VEC (probe_ops_cp) *all_probe_ops;

/* The probe_ops associated with the generic probe.  */

extern const struct probe_ops probe_ops_any;

/* Helper function that, given KEYWORDS, iterate over it trying to match
   each keyword with LINESPECP.  If it succeeds, it updates the LINESPECP
   pointer and returns 1.  Otherwise, nothing is done to LINESPECP and zero
   is returned.  */

extern int probe_is_linespec_by_keyword (const char **linespecp,
					 const char *const *keywords);

/* Return specific PROBE_OPS * matching *LINESPECP and possibly updating
   *LINESPECP to skip its "-probe-type " prefix.  Return &probe_ops_any if
   *LINESPECP matches "-probe ", that is any unspecific probe.  Return NULL if
   *LINESPECP is not identified as any known probe type, *LINESPECP is not
   modified in such case.  */

extern const struct probe_ops *probe_linespec_to_ops (const char **linespecp);

/* The probe itself.  The struct contains generic information about the
   probe, and then some specific information which should be stored in
   the `probe_info' field.  */

struct probe
  {
    /* The operations associated with this probe.  */
    const struct probe_ops *pops;

    /* The objfile which contains this probe.  Even if the probe is also
       present in a separate debug objfile, this variable always points to
       the non-separate debug objfile.  */
    struct objfile *objfile;

    /* The name of the probe.  */
    const char *name;

    /* The provider of the probe.  It generally defaults to the name of
       the objfile which contains the probe.  */
    const char *provider;

    /* The address where the probe is inserted.  */
    CORE_ADDR address;
  };

/* A helper for linespec that decodes a probe specification.  It returns a
   symtabs_and_lines object and updates *ARGPTR or throws an error.  The
   argument PTYPE specifies the type of the probe(s) to be parsed.  */

extern struct symtabs_and_lines parse_probes (char **argptr,
					      struct linespec_result *canon);

/* Helper function to register the proper probe_ops to a newly created probe.
   This function is mainly called from `sym_get_probes'.  */

extern void register_probe_ops (struct probe *probe);

/* Given a PC, find an associated probe with type PTYPE.  If a probe is
   found, return it.  If no probe is found, return NULL.  */

extern struct probe *find_probe_by_pc (CORE_ADDR pc);

/* Search OBJFILE for a probe with the given PROVIDER, NAME and PTYPE.
   Return a VEC of all probes that were found.  If no matching probe
   is found, return NULL.  The caller must free the VEC.  */

extern VEC (probe_p) *find_probes_in_objfile (struct objfile *objfile,
					      const char *provider,
					      const char *name);

/* Generate a `info probes' command output for probe_ops represented by
   POPS.  If POPS is NULL it considers any probes types.  It is a helper
   function that can be used by the probe backends to print their
   `info probe TYPE'.  */

extern void info_probes_for_ops (char *arg, int from_tty,
				 const struct probe_ops *pops);

/* Return the `cmd_list_element' associated with the `info probes' command,
   or create a new one if it doesn't exist.  Helper function that serves the
   purpose of avoiding the case of a backend using the `cmd_list_element'
   associated with `info probes', without having it registered yet.  */

extern struct cmd_list_element **info_probes_cmdlist_get (void);

/* A convenience function that finds a probe at the PC in FRAME and
   evaluates argument N, with 0 <= N < number_of_args.  If there is no
   probe at that location, or if the probe does not have enough arguments,
   this returns NULL.  */

extern struct value *probe_safe_evaluate_at_pc (struct frame_info *frame,
						unsigned n);

#endif /* !defined (PROBE_H) */
