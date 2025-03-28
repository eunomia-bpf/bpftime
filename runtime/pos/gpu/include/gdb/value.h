/* Definitions for values of C expressions, for GDB.

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

#if !defined (VALUE_H)
#define VALUE_H 1

#include "doublest.h"
#include "frame.h"		/* For struct frame_id.  */

/* The maximum number of array elements to read when fetching the array from 
   the target. Set to print_max by printcmd.c and stack.c before calling some 
   valops functions, then reset. 0 represents 'unlimited'. Used in 
   get_limited_length for limiting array fetches - not necessarily structures 
   with large array members for example. */
int read_element_limit;

unsigned int get_limited_length(struct type *type);

struct block;
struct expression;
struct regcache;
struct symbol;
struct type;
struct ui_file;
struct language_defn;
struct value_print_options;

/* The structure which defines the type of a value.  It should never
   be possible for a program lval value to survive over a call to the
   inferior (i.e. to be put into the history list or an internal
   variable).  */

struct value;

/* Values are stored in a chain, so that they can be deleted easily
   over calls to the inferior.  Values assigned to internal variables,
   put into the value history or exposed to Python are taken off this
   list.  */

struct value *value_next (struct value *);

/* Type of the value.  */

extern struct type *value_type (const struct value *);

/* This is being used to change the type of an existing value, that
   code should instead be creating a new value with the changed type
   (but possibly shared content).  */

extern void deprecated_set_value_type (struct value *value,
				       struct type *type);

/* Only used for bitfields; number of bits contained in them.  */

extern int value_bitsize (const struct value *);
extern void set_value_bitsize (struct value *, int bit);

/* Only used for bitfields; position of start of field.  For
   gdbarch_bits_big_endian=0 targets, it is the position of the LSB.  For
   gdbarch_bits_big_endian=1 targets, it is the position of the MSB.  */

extern int value_bitpos (const struct value *);
extern void set_value_bitpos (struct value *, int bit);

/* Only used for bitfields; the containing value.  This allows a
   single read from the target when displaying multiple
   bitfields.  */

struct value *value_parent (struct value *);
extern void set_value_parent (struct value *value, struct value *parent);

/* Describes offset of a value within lval of a structure in bytes.
   If lval == lval_memory, this is an offset to the address.  If lval
   == lval_register, this is a further offset from location.address
   within the registers structure.  Note also the member
   embedded_offset below.  */

extern int value_offset (const struct value *);
extern void set_value_offset (struct value *, int offset);

/* The comment from "struct value" reads: ``Is it modifiable?  Only
   relevant if lval != not_lval.''.  Shouldn't the value instead be
   not_lval and be done with it?  */

extern int deprecated_value_modifiable (struct value *value);

/* If a value represents a C++ object, then the `type' field gives the
   object's compile-time type.  If the object actually belongs to some
   class derived from `type', perhaps with other base classes and
   additional members, then `type' is just a subobject of the real
   thing, and the full object is probably larger than `type' would
   suggest.

   If `type' is a dynamic class (i.e. one with a vtable), then GDB can
   actually determine the object's run-time type by looking at the
   run-time type information in the vtable.  When this information is
   available, we may elect to read in the entire object, for several
   reasons:

   - When printing the value, the user would probably rather see the
     full object, not just the limited portion apparent from the
     compile-time type.

   - If `type' has virtual base classes, then even printing `type'
     alone may require reaching outside the `type' portion of the
     object to wherever the virtual base class has been stored.

   When we store the entire object, `enclosing_type' is the run-time
   type -- the complete object -- and `embedded_offset' is the offset
   of `type' within that larger type, in bytes.  The value_contents()
   macro takes `embedded_offset' into account, so most GDB code
   continues to see the `type' portion of the value, just as the
   inferior would.

   If `type' is a pointer to an object, then `enclosing_type' is a
   pointer to the object's run-time type, and `pointed_to_offset' is
   the offset in bytes from the full object to the pointed-to object
   -- that is, the value `embedded_offset' would have if we followed
   the pointer and fetched the complete object.  (I don't really see
   the point.  Why not just determine the run-time type when you
   indirect, and avoid the special case?  The contents don't matter
   until you indirect anyway.)

   If we're not doing anything fancy, `enclosing_type' is equal to
   `type', and `embedded_offset' is zero, so everything works
   normally.  */

extern struct type *value_enclosing_type (struct value *);
extern void set_value_enclosing_type (struct value *val,
				      struct type *new_type);

/* Returns value_type or value_enclosing_type depending on
   value_print_options.objectprint.

   If RESOLVE_SIMPLE_TYPES is 0 the enclosing type will be resolved
   only for pointers and references, else it will be returned
   for all the types (e.g. structures).  This option is useful
   to prevent retrieving enclosing type for the base classes fields.

   REAL_TYPE_FOUND is used to inform whether the real type was found
   (or just static type was used).  The NULL may be passed if it is not
   necessary. */

extern struct type *value_actual_type (struct value *value,
				       int resolve_simple_types,
				       int *real_type_found);

extern int value_pointed_to_offset (struct value *value);
extern void set_value_pointed_to_offset (struct value *value, int val);
extern int value_embedded_offset (struct value *value);
extern void set_value_embedded_offset (struct value *value, int val);

/* For lval_computed values, this structure holds functions used to
   retrieve and set the value (or portions of the value).

   For each function, 'V' is the 'this' pointer: an lval_funcs
   function F may always assume that the V it receives is an
   lval_computed value, and has F in the appropriate slot of its
   lval_funcs structure.  */

struct lval_funcs
{
  /* Fill in VALUE's contents.  This is used to "un-lazy" values.  If
     a problem arises in obtaining VALUE's bits, this function should
     call 'error'.  If it is NULL value_fetch_lazy on "un-lazy"
     non-optimized-out value is an internal error.  */
  void (*read) (struct value *v);

  /* Handle an assignment TOVAL = FROMVAL by writing the value of
     FROMVAL to TOVAL's location.  The contents of TOVAL have not yet
     been updated.  If a problem arises in doing so, this function
     should call 'error'.  If it is NULL such TOVAL assignment is an error as
     TOVAL is not considered as an lvalue.  */
  void (*write) (struct value *toval, struct value *fromval);

  /* Check the validity of some bits in VALUE.  This should return 1
     if all the bits starting at OFFSET and extending for LENGTH bits
     are valid, or 0 if any bit is invalid.  */
  int (*check_validity) (const struct value *value, int offset, int length);

  /* Return 1 if any bit in VALUE is valid, 0 if they are all invalid.  */
  int (*check_any_valid) (const struct value *value);

  /* If non-NULL, this is used to implement pointer indirection for
     this value.  This method may return NULL, in which case value_ind
     will fall back to ordinary indirection.  */
  struct value *(*indirect) (struct value *value);

  /* If non-NULL, this is used to implement reference resolving for
     this value.  This method may return NULL, in which case coerce_ref
     will fall back to ordinary references resolving.  */
  struct value *(*coerce_ref) (const struct value *value);

  /* If non-NULL, this is used to determine whether the indicated bits
     of VALUE are a synthetic pointer.  */
  int (*check_synthetic_pointer) (const struct value *value,
				  int offset, int length);

  /* Return a duplicate of VALUE's closure, for use in a new value.
     This may simply return the same closure, if VALUE's is
     reference-counted or statically allocated.

     This may be NULL, in which case VALUE's closure is re-used in the
     new value.  */
  void *(*copy_closure) (const struct value *v);

  /* Drop VALUE's reference to its closure.  Maybe this frees the
     closure; maybe this decrements a reference count; maybe the
     closure is statically allocated and this does nothing.

     This may be NULL, in which case no action is taken to free
     VALUE's closure.  */
  void (*free_closure) (struct value *v);
};

/* Create a computed lvalue, with type TYPE, function pointers FUNCS,
   and closure CLOSURE.  */

extern struct value *allocate_computed_value (struct type *type,
					      const struct lval_funcs *funcs,
					      void *closure);

/* Helper function to check the validity of some bits of a value.

   If TYPE represents some aggregate type (e.g., a structure), return 1.
   
   Otherwise, any of the bytes starting at OFFSET and extending for
   TYPE_LENGTH(TYPE) bytes are invalid, print a message to STREAM and
   return 0.  The checking is done using FUNCS.
   
   Otherwise, return 1.  */

extern int valprint_check_validity (struct ui_file *stream, struct type *type,
				    int embedded_offset,
				    const struct value *val);

extern struct value *allocate_optimized_out_value (struct type *type);

/* If VALUE is lval_computed, return its lval_funcs structure.  */

extern const struct lval_funcs *value_computed_funcs (const struct value *);

/* If VALUE is lval_computed, return its closure.  The meaning of the
   returned value depends on the functions VALUE uses.  */

extern void *value_computed_closure (const struct value *value);

/* If zero, contents of this value are in the contents field.  If
   nonzero, contents are in inferior.  If the lval field is lval_memory,
   the contents are in inferior memory at location.address plus offset.
   The lval field may also be lval_register.

   WARNING: This field is used by the code which handles watchpoints
   (see breakpoint.c) to decide whether a particular value can be
   watched by hardware watchpoints.  If the lazy flag is set for some
   member of a value chain, it is assumed that this member of the
   chain doesn't need to be watched as part of watching the value
   itself.  This is how GDB avoids watching the entire struct or array
   when the user wants to watch a single struct member or array
   element.  If you ever change the way lazy flag is set and reset, be
   sure to consider this use as well!  */

extern int value_lazy (struct value *);
extern void set_value_lazy (struct value *value, int val);

extern int value_stack (struct value *);
extern void set_value_stack (struct value *value, int val);

/* value_contents() and value_contents_raw() both return the address
   of the gdb buffer used to hold a copy of the contents of the lval.
   value_contents() is used when the contents of the buffer are needed
   -- it uses value_fetch_lazy() to load the buffer from the process
   being debugged if it hasn't already been loaded
   (value_contents_writeable() is used when a writeable but fetched
   buffer is required)..  value_contents_raw() is used when data is
   being stored into the buffer, or when it is certain that the
   contents of the buffer are valid.

   Note: The contents pointer is adjusted by the offset required to
   get to the real subobject, if the value happens to represent
   something embedded in a larger run-time object.  */

extern gdb_byte *value_contents_raw (struct value *);

/* Actual contents of the value.  For use of this value; setting it
   uses the stuff above.  Not valid if lazy is nonzero.  Target
   byte-order.  We force it to be aligned properly for any possible
   value.  Note that a value therefore extends beyond what is
   declared here.  */

extern const gdb_byte *value_contents (struct value *);
extern gdb_byte *value_contents_writeable (struct value *);

/* The ALL variants of the above two macros do not adjust the returned
   pointer by the embedded_offset value.  */

extern gdb_byte *value_contents_all_raw (struct value *);
extern const gdb_byte *value_contents_all (struct value *);

/* Like value_contents_all, but does not require that the returned
   bits be valid.  This should only be used in situations where you
   plan to check the validity manually.  */
extern const gdb_byte *value_contents_for_printing (struct value *value);

/* Like value_contents_for_printing, but accepts a constant value
   pointer.  Unlike value_contents_for_printing however, the pointed
   value must _not_ be lazy.  */
extern const gdb_byte *
  value_contents_for_printing_const (const struct value *value);

extern int value_fetch_lazy (struct value *val);
extern int value_contents_equal (struct value *val1, struct value *val2);

/* If nonzero, this is the value of a variable which does not actually
   exist in the program.  */
extern int value_optimized_out (struct value *value);
extern void set_value_optimized_out (struct value *value, int val);

/* CUDA - register cache */
extern int value_cached (const struct value *value);
extern void set_value_cached (struct value *value, int val);
/* CUDA - regmap extrapolation */
extern int value_extrapolated (const struct value *value);
extern void set_value_extrapolated (struct value *value, int val);

/* Like value_optimized_out, but return false if any bit in the object
   is valid.  */
extern int value_entirely_optimized_out (const struct value *value);

/* Set or return field indicating whether a variable is initialized or
   not, based on debugging information supplied by the compiler.
   1 = initialized; 0 = uninitialized.  */
extern int value_initialized (struct value *);
extern void set_value_initialized (struct value *, int);

/* Set COMPONENT's location as appropriate for a component of WHOLE
   --- regardless of what kind of lvalue WHOLE is.  */
extern void set_value_component_location (struct value *component,
                                          const struct value *whole);

/* While the following fields are per- VALUE .CONTENT .PIECE (i.e., a
   single value might have multiple LVALs), this hacked interface is
   limited to just the first PIECE.  Expect further change.  */
/* Type of value; either not an lval, or one of the various different
   possible kinds of lval.  */
extern enum lval_type *deprecated_value_lval_hack (struct value *);
#define VALUE_LVAL(val) (*deprecated_value_lval_hack (val))

/* Like VALUE_LVAL, except the parameter can be const.  */
extern enum lval_type value_lval_const (const struct value *value);

/* If lval == lval_memory, return the address in the inferior.  If
   lval == lval_register, return the byte offset into the registers
   structure.  Otherwise, return 0.  The returned address
   includes the offset, if any.  */
extern CORE_ADDR value_address (const struct value *);

/* Like value_address, except the result does not include value's
   offset.  */
extern CORE_ADDR value_raw_address (struct value *);

/* Set the address of a value.  */
extern void set_value_address (struct value *, CORE_ADDR);

/* Pointer to internal variable.  */
extern struct internalvar **deprecated_value_internalvar_hack (struct value *);
#define VALUE_INTERNALVAR(val) (*deprecated_value_internalvar_hack (val))

/* Frame register value is relative to.  This will be described in the
   lval enum above as "lval_register".  */
extern struct frame_id *deprecated_value_frame_id_hack (struct value *);
#define VALUE_FRAME_ID(val) (*deprecated_value_frame_id_hack (val))

/* Register number if the value is from a register.  */
extern short *deprecated_value_regnum_hack (struct value *);
#define VALUE_REGNUM(val) (*deprecated_value_regnum_hack (val))

/* Return value after lval_funcs->coerce_ref (after check_typedef).  Return
   NULL if lval_funcs->coerce_ref is not applicable for whatever reason.  */

extern struct value *coerce_ref_if_computed (const struct value *arg);

/* Setup a new value type and enclosing value type for dereferenced value VALUE.
   ENC_TYPE is the new enclosing type that should be set.  ORIGINAL_TYPE and
   ORIGINAL_VAL are the type and value of the original reference or pointer.

   Note, that VALUE is modified by this function.

   It is a common implementation for coerce_ref and value_ind.  */

extern struct value * readjust_indirect_value_type (struct value *value,
						    struct type *enc_type,
						    struct type *original_type,
						    struct value *original_val);

/* Convert a REF to the object referenced.  */

extern struct value *coerce_ref (struct value *value);

/* If ARG is an array, convert it to a pointer.
   If ARG is a function, convert it to a function pointer.

   References are dereferenced.  */

extern struct value *coerce_array (struct value *value);

/* Given a value, determine whether the bits starting at OFFSET and
   extending for LENGTH bits are valid.  This returns nonzero if all
   bits in the given range are valid, zero if any bit is invalid.  */

extern int value_bits_valid (const struct value *value,
			     int offset, int length);

/* Given a value, determine whether the bits starting at OFFSET and
   extending for LENGTH bits are a synthetic pointer.  */

extern int value_bits_synthetic_pointer (const struct value *value,
					 int offset, int length);

/* Given a value, determine whether the contents bytes starting at
   OFFSET and extending for LENGTH bytes are available.  This returns
   nonzero if all bytes in the given range are available, zero if any
   byte is unavailable.  */

extern int value_bytes_available (const struct value *value,
				  int offset, int length);

/* Like value_bytes_available, but return false if any byte in the
   whole object is unavailable.  */
extern int value_entirely_available (struct value *value);

/* Mark VALUE's content bytes starting at OFFSET and extending for
   LENGTH bytes as unavailable.  */

extern void mark_value_bytes_unavailable (struct value *value,
					  int offset, int length);

/* Compare LENGTH bytes of VAL1's contents starting at OFFSET1 with
   LENGTH bytes of VAL2's contents starting at OFFSET2.

   Note that "contents" refers to the whole value's contents
   (value_contents_all), without any embedded offset adjustment.  For
   example, to compare a complete object value with itself, including
   its enclosing type chunk, you'd do:

     int len = TYPE_LENGTH (check_typedef (value_enclosing_type (val)));
     value_available_contents (val, 0, val, 0, len);

   Returns true iff the set of available contents match.  Unavailable
   contents compare equal with unavailable contents, and different
   with any available byte.  For example, if 'x's represent an
   unavailable byte, and 'V' and 'Z' represent different available
   bytes, in a value with length 16:

   offset:   0   4   8   12  16
   contents: xxxxVVVVxxxxVVZZ

   then:

   value_available_contents_eq(val, 0, val, 8, 6) => 1
   value_available_contents_eq(val, 0, val, 4, 4) => 1
   value_available_contents_eq(val, 0, val, 8, 8) => 0
   value_available_contents_eq(val, 4, val, 12, 2) => 1
   value_available_contents_eq(val, 4, val, 12, 4) => 0
   value_available_contents_eq(val, 3, val, 4, 4) => 0

   We only know whether a value chunk is available if we've tried to
   read it.  As this routine is used by printing routines, which may
   be printing values in the value history, long after the inferior is
   gone, it works with const values.  Therefore, this routine must not
   be called with lazy values.  */

extern int value_available_contents_eq (const struct value *val1, int offset1,
					const struct value *val2, int offset2,
					int length);

/* Read LENGTH bytes of memory starting at MEMADDR into BUFFER, which
   is (or will be copied to) VAL's contents buffer offset by
   EMBEDDED_OFFSET (that is, to &VAL->contents[EMBEDDED_OFFSET]).
   Marks value contents ranges as unavailable if the corresponding
   memory is likewise unavailable.  STACK indicates whether the memory
   is known to be stack memory.  */

extern void read_value_memory (struct value *val, int embedded_offset,
			       int stack, CORE_ADDR memaddr,
			       gdb_byte *buffer, size_t length);

/* Cast SCALAR_VALUE to the element type of VECTOR_TYPE, then replicate
   into each element of a new vector value with VECTOR_TYPE.  */

struct value *value_vector_widen (struct value *scalar_value,
				  struct type *vector_type);



#include "symtab.h"
#include "gdbtypes.h"
#include "expression.h"

struct frame_info;
struct fn_field;

extern int print_address_demangle (const struct value_print_options *,
				   struct gdbarch *, CORE_ADDR,
				   struct ui_file *, int);

extern LONGEST value_as_long (struct value *val);
extern DOUBLEST value_as_double (struct value *val);
extern CORE_ADDR value_as_address (struct value *val);

extern LONGEST unpack_long (struct type *type, const gdb_byte *valaddr);
extern DOUBLEST unpack_double (struct type *type, const gdb_byte *valaddr,
			       int *invp);
extern CORE_ADDR unpack_pointer (struct type *type, const gdb_byte *valaddr);

extern int unpack_value_bits_as_long (struct type *field_type,
				      const gdb_byte *valaddr,
				      int embedded_offset, int bitpos,
				      int bitsize,
				      const struct value *original_value,
				      LONGEST *result);

extern LONGEST unpack_field_as_long (struct type *type,
				     const gdb_byte *valaddr,
				     int fieldno);
extern int unpack_value_field_as_long (struct type *type, const gdb_byte *valaddr,
				int embedded_offset, int fieldno,
				const struct value *val, LONGEST *result);

extern struct value *value_field_bitfield (struct type *type, int fieldno,
					   const gdb_byte *valaddr,
					   int embedded_offset,
					   const struct value *val);

extern void pack_long (gdb_byte *buf, struct type *type, LONGEST num);

extern struct value *value_from_longest (struct type *type, LONGEST num);
extern struct value *value_from_ulongest (struct type *type, ULONGEST num);
extern struct value *value_from_pointer (struct type *type, CORE_ADDR addr);
extern struct value *value_from_double (struct type *type, DOUBLEST num);
extern struct value *value_from_decfloat (struct type *type,
					  const gdb_byte *decbytes);
extern struct value *value_from_history_ref (char *, char **);

extern struct value *value_at (struct type *type, CORE_ADDR addr);
extern struct value *value_at_lazy (struct type *type, CORE_ADDR addr);

extern struct value *value_from_contents_and_address (struct type *,
						      const gdb_byte *,
						      unsigned length,
						      CORE_ADDR);
extern struct value *value_from_contents (struct type *, const gdb_byte *);

extern struct value *default_value_from_register (struct type *type,
						  int regnum,
						  struct frame_info *frame);

extern void read_frame_register_value (struct value *value,
				       struct frame_info *frame);

extern struct value *value_from_register (struct type *type, int regnum,
					  struct frame_info *frame);

extern CORE_ADDR address_from_register (struct type *type, int regnum,
					struct frame_info *frame);

extern struct value *value_of_variable (struct symbol *var,
					const struct block *b);

extern struct value *address_of_variable (struct symbol *var,
					  const struct block *b);

extern struct value *value_of_register (int regnum, struct frame_info *frame);

struct value *value_of_register_lazy (struct frame_info *frame, int regnum);

extern int symbol_read_needs_frame (struct symbol *);

extern struct value *read_var_value (struct symbol *var,
				     struct frame_info *frame);

extern struct value *default_read_var_value (struct symbol *var,
					     struct frame_info *frame);

extern struct value *allocate_value (struct type *type);
extern struct value *allocate_value_lazy (struct type *type);
extern void allocate_value_contents (struct value *value);
extern void value_contents_copy (struct value *dst, int dst_offset,
				 struct value *src, int src_offset,
				 int length);
extern void value_contents_copy_raw (struct value *dst, int dst_offset,
				     struct value *src, int src_offset,
				     int length);

extern struct value *allocate_repeat_value (struct type *type, int count);

extern struct value *value_mark (void);

extern void value_free_to_mark (struct value *mark);

extern struct value *value_cstring (char *ptr, ssize_t len,
				    struct type *char_type);
extern struct value *value_string (char *ptr, ssize_t len,
				   struct type *char_type);

extern struct value *value_array (int lowbound, int highbound,
				  struct value **elemvec);

extern struct value *value_concat (struct value *arg1, struct value *arg2);

extern struct value *value_binop (struct value *arg1, struct value *arg2,
				  enum exp_opcode op);

extern struct value *value_ptradd (struct value *arg1, LONGEST arg2);

extern LONGEST value_ptrdiff (struct value *arg1, struct value *arg2);

extern int value_must_coerce_to_target (struct value *arg1);

extern struct value *value_coerce_to_target (struct value *arg1);

extern struct value *value_coerce_array (struct value *arg1);

extern struct value *value_coerce_function (struct value *arg1);

extern struct value *value_ind (struct value *arg1);

extern struct value *value_addr (struct value *arg1);

extern struct value *value_ref (struct value *arg1);

extern struct value *value_assign (struct value *toval,
				   struct value *fromval);

extern struct value *value_pos (struct value *arg1);

extern struct value *value_neg (struct value *arg1);

extern struct value *value_complement (struct value *arg1);

extern struct value *value_struct_elt (struct value **argp,
				       struct value **args,
				       const char *name, int *static_memfuncp,
				       const char *err);

extern struct value *value_aggregate_elt (struct type *curtype,
					  char *name,
					  struct type *expect_type,
					  int want_address,
					  enum noside noside);

extern struct value *value_static_field (struct type *type, int fieldno);

enum oload_search_type { NON_METHOD, METHOD, BOTH };

extern int find_overload_match (struct value **args, int nargs,
				const char *name,
				enum oload_search_type method,
				struct value **objp, struct symbol *fsym,
				struct value **valp, struct symbol **symp,
				int *staticp, const int no_adl);

extern struct value *value_field (struct value *arg1, int fieldno);

extern struct value *value_primitive_field (struct value *arg1, int offset,
					    int fieldno,
					    struct type *arg_type);


extern struct type *value_rtti_indirect_type (struct value *, int *, int *,
					      int *);

extern struct value *value_full_object (struct value *, struct type *, int,
					int, int);

extern struct value *value_cast_pointers (struct type *, struct value *, int);

extern struct value *value_cast (struct type *type, struct value *arg2);

extern struct value *value_reinterpret_cast (struct type *type,
					     struct value *arg);

extern struct value *value_dynamic_cast (struct type *type, struct value *arg);

extern struct value *value_zero (struct type *type, enum lval_type lv);

extern struct value *value_one (struct type *type);

extern struct value *value_repeat (struct value *arg1, int count);

extern struct value *value_subscript (struct value *array, LONGEST index);

extern struct value *value_bitstring_subscript (struct type *type,
						struct value *bitstring,
						LONGEST index);

extern struct value *register_value_being_returned (struct type *valtype,
						    struct regcache *retbuf);

extern int value_in (struct value *element, struct value *set);

extern int value_bit_index (struct type *type, const gdb_byte *addr,
			    int index);

extern enum return_value_convention
struct_return_convention (struct gdbarch *gdbarch, struct value *function,
			  struct type *value_type);

extern int using_struct_return (struct gdbarch *gdbarch,
				struct value *function,
				struct type *value_type);

extern struct value *evaluate_expression (struct expression *exp);

extern struct value *evaluate_type (struct expression *exp);

extern struct value *evaluate_subexp (struct type *expect_type,
				      struct expression *exp,
				      int *pos, enum noside noside);

extern struct value *evaluate_subexpression_type (struct expression *exp,
						  int subexp);

extern void fetch_subexp_value (struct expression *exp, int *pc,
				struct value **valp, struct value **resultp,
				struct value **val_chain);

extern char *extract_field_op (struct expression *exp, int *subexp);

extern struct value *evaluate_subexp_with_coercion (struct expression *,
						    int *, enum noside);

extern struct value *parse_and_eval (const char *exp);

extern struct value *parse_to_comma_and_eval (const char **expp);

extern struct type *parse_and_eval_type (char *p, int length);

extern CORE_ADDR parse_and_eval_address (const char *exp);

extern LONGEST parse_and_eval_long (char *exp);

extern void unop_promote (const struct language_defn *language,
			  struct gdbarch *gdbarch,
			  struct value **arg1);

extern void binop_promote (const struct language_defn *language,
			   struct gdbarch *gdbarch,
			   struct value **arg1, struct value **arg2);

extern struct value *access_value_history (int num);

extern struct value *value_of_internalvar (struct gdbarch *gdbarch,
					   struct internalvar *var);

extern int get_internalvar_integer (struct internalvar *var, LONGEST *l);

extern void set_internalvar (struct internalvar *var, struct value *val);

extern void set_internalvar_integer (struct internalvar *var, LONGEST l);

extern void set_internalvar_string (struct internalvar *var,
				    const char *string);

extern void clear_internalvar (struct internalvar *var);

extern void set_internalvar_component (struct internalvar *var,
				       int offset,
				       int bitpos, int bitsize,
				       struct value *newvalue);

extern struct internalvar *lookup_only_internalvar (const char *name);

extern struct internalvar *create_internalvar (const char *name);

extern VEC (char_ptr) *complete_internalvar (const char *name);

/* An internalvar can be dynamically computed by supplying a vector of
   function pointers to perform various operations.  */

struct internalvar_funcs
{
  /* Compute the value of the variable.  The DATA argument passed to
     the function is the same argument that was passed to
     `create_internalvar_type_lazy'.  */

  struct value *(*make_value) (struct gdbarch *arch,
			       struct internalvar *var,
			       void *data);

  /* Update the agent expression EXPR with bytecode to compute the
     value.  VALUE is the agent value we are updating.  The DATA
     argument passed to this function is the same argument that was
     passed to `create_internalvar_type_lazy'.  If this pointer is
     NULL, then the internalvar cannot be compiled to an agent
     expression.  */

  void (*compile_to_ax) (struct internalvar *var,
			 struct agent_expr *expr,
			 struct axs_value *value,
			 void *data);

  /* If non-NULL, this is called to destroy DATA.  The DATA argument
     passed to this function is the same argument that was passed to
     `create_internalvar_type_lazy'.  */

  void (*destroy) (void *data);
};

extern struct internalvar *
create_internalvar_type_lazy (const char *name,
			      const struct internalvar_funcs *funcs,
			      void *data);

/* Compile an internal variable to an agent expression.  VAR is the
   variable to compile; EXPR and VALUE are the agent expression we are
   updating.  This will return 0 if there is no known way to compile
   VAR, and 1 if VAR was successfully compiled.  It may also throw an
   exception on error.  */

extern int compile_internalvar_to_ax (struct internalvar *var,
				      struct agent_expr *expr,
				      struct axs_value *value);

extern struct internalvar *lookup_internalvar (const char *name);

extern int value_equal (struct value *arg1, struct value *arg2);

extern int value_equal_contents (struct value *arg1, struct value *arg2);

extern int value_less (struct value *arg1, struct value *arg2);

extern int value_logical_not (struct value *arg1);

/* C++ */

extern struct value *value_of_this (const struct language_defn *lang);

extern struct value *value_of_this_silent (const struct language_defn *lang);

extern struct value *value_x_binop (struct value *arg1, struct value *arg2,
				    enum exp_opcode op,
				    enum exp_opcode otherop,
				    enum noside noside);

extern struct value *value_x_unop (struct value *arg1, enum exp_opcode op,
				   enum noside noside);

extern struct value *value_fn_field (struct value **arg1p, struct fn_field *f,
				     int j, struct type *type, int offset);

extern int binop_types_user_defined_p (enum exp_opcode op,
				       struct type *type1,
				       struct type *type2);

extern int binop_user_defined_p (enum exp_opcode op, struct value *arg1,
				 struct value *arg2);

extern int unop_user_defined_p (enum exp_opcode op, struct value *arg1);

extern int destructor_name_p (const char *name, struct type *type);

extern void value_incref (struct value *val);

extern void value_free (struct value *val);

extern void free_all_values (void);

extern void free_value_chain (struct value *v);

extern void release_value (struct value *val);

extern void release_value_or_incref (struct value *val);

extern int record_latest_value (struct value *val);

extern void modify_field (struct type *type, gdb_byte *addr,
			  LONGEST fieldval, int bitpos, int bitsize);

extern void type_print (struct type *type, const char *varstring,
			struct ui_file *stream, int show);

extern char *type_to_string (struct type *type);

extern gdb_byte *baseclass_addr (struct type *type, int index,
				 gdb_byte *valaddr,
				 struct value **valuep, int *errp);

extern void print_longest (struct ui_file *stream, int format,
			   int use_local, LONGEST val);

extern void print_floating (const gdb_byte *valaddr, struct type *type,
			    struct ui_file *stream);

extern void print_decimal_floating (const gdb_byte *valaddr, struct type *type,
				    struct ui_file *stream);

extern void value_print (struct value *val, struct ui_file *stream,
			 const struct value_print_options *options);

extern void value_print_array_elements (struct value *val,
					struct ui_file *stream, int format,
					enum val_prettyprint pretty);

extern struct value *value_release_to_mark (struct value *mark);

extern void val_print (struct type *type, const gdb_byte *valaddr,
		       int embedded_offset, CORE_ADDR address,
		       struct ui_file *stream, int recurse,
		       const struct value *val,
		       const struct value_print_options *options,
		       const struct language_defn *language);

extern void common_val_print (struct value *val,
			      struct ui_file *stream, int recurse,
			      const struct value_print_options *options,
			      const struct language_defn *language);

extern int val_print_string (struct type *elttype, const char *encoding,
			     CORE_ADDR addr, int len,
			     struct ui_file *stream,
			     const struct value_print_options *options);

extern void print_variable_and_value (const char *name,
				      struct symbol *var,
				      struct frame_info *frame,
				      struct ui_file *stream,
				      int indent);

extern void typedef_print (struct type *type, struct symbol *news,
			   struct ui_file *stream);

extern char *internalvar_name (struct internalvar *var);

extern void preserve_values (struct objfile *);

/* From values.c */

extern struct value *value_copy (struct value *);

extern struct value *value_non_lval (struct value *);

extern void preserve_one_value (struct value *, struct objfile *, htab_t);

/* From valops.c */

extern struct value *varying_to_slice (struct value *);

extern struct value *value_slice (struct value *, int, int);

extern struct value *value_literal_complex (struct value *, struct value *,
					    struct type *);

extern struct value *value_real (struct value *);

extern struct value *value_imag (struct value *);

extern struct value *find_function_in_inferior (const char *,
						struct objfile **);

extern struct value *value_allocate_space_in_inferior (int);

extern struct value *value_subscripted_rvalue (struct value *array,
					       LONGEST index, int lowerbound);

/* User function handler.  */

typedef struct value *(*internal_function_fn) (struct gdbarch *gdbarch,
					       const struct language_defn *language,
					       void *cookie,
					       int argc,
					       struct value **argv);

void add_internal_function (const char *name, const char *doc,
			    internal_function_fn handler,
			    void *cookie);

struct value *call_internal_function (struct gdbarch *gdbarch,
				      const struct language_defn *language,
				      struct value *function,
				      int argc, struct value **argv);

char *value_internal_function_name (struct value *);

extern const gdb_byte * value_contents_all_safe (struct value *value);

extern int value_repeated (const struct value *value);
extern void set_value_repeated (struct value *value,
				int repeated);

extern unsigned value_length (const struct value *value);

extern void value_copy_contents (struct value *to, struct value *from);
extern void value_copy_contents_all_raw (struct value *to, struct value *from);

#endif /* !defined (VALUE_H) */
