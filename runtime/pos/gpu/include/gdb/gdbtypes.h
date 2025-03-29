/* Internal type definitions for GDB.

   Copyright (C) 1992-2013 Free Software Foundation, Inc.

   Contributed by Cygnus Support, using pieces from other GDB modules.

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

#if !defined (GDBTYPES_H)
#define GDBTYPES_H 1

#include "hashtab.h"

/* Forward declarations for prototypes.  */
struct field;
struct block;
struct value_print_options;
struct language_defn;

/* These declarations are DWARF-specific as some of the gdbtypes.h data types
   are already DWARF-specific.  */

/* Offset relative to the start of its containing CU (compilation unit).  */
typedef struct
{
  unsigned int cu_off;
} cu_offset;

/* Offset relative to the start of its .debug_info or .debug_types section.  */
typedef struct
{
  unsigned int sect_off;
} sect_offset;

/* Some macros for char-based bitfields.  */

#define B_SET(a,x)	((a)[(x)>>3] |= (1 << ((x)&7)))
#define B_CLR(a,x)	((a)[(x)>>3] &= ~(1 << ((x)&7)))
#define B_TST(a,x)	((a)[(x)>>3] & (1 << ((x)&7)))
#define B_TYPE		unsigned char
#define	B_BYTES(x)	( 1 + ((x)>>3) )
#define	B_CLRALL(a,x)	memset ((a), 0, B_BYTES(x))

/* Different kinds of data types are distinguished by the `code' field.  */

enum type_code
  {
    TYPE_CODE_BITSTRING = -1,	/* Deprecated  */
    TYPE_CODE_UNDEF = 0,	/* Not used; catches errors */
    TYPE_CODE_PTR,		/* Pointer type */

    /* Array type with lower & upper bounds.

       Regardless of the language, GDB represents multidimensional
       array types the way C does: as arrays of arrays.  So an
       instance of a GDB array type T can always be seen as a series
       of instances of TYPE_TARGET_TYPE (T) laid out sequentially in
       memory.

       Row-major languages like C lay out multi-dimensional arrays so
       that incrementing the rightmost index in a subscripting
       expression results in the smallest change in the address of the
       element referred to.  Column-major languages like Fortran lay
       them out so that incrementing the leftmost index results in the
       smallest change.

       This means that, in column-major languages, working our way
       from type to target type corresponds to working through indices
       from right to left, not left to right.  */
    TYPE_CODE_ARRAY,

    TYPE_CODE_STRUCT,		/* C struct or Pascal record */
    TYPE_CODE_UNION,		/* C union or Pascal variant part */
    TYPE_CODE_ENUM,		/* Enumeration type */
    TYPE_CODE_FLAGS,		/* Bit flags type */
    TYPE_CODE_FUNC,		/* Function type */
    TYPE_CODE_INT,		/* Integer type */

    /* Floating type.  This is *NOT* a complex type.  Beware, there are parts
       of GDB which bogusly assume that TYPE_CODE_FLT can mean complex.  */
    TYPE_CODE_FLT,

    /* Void type.  The length field specifies the length (probably always
       one) which is used in pointer arithmetic involving pointers to
       this type, but actually dereferencing such a pointer is invalid;
       a void type has no length and no actual representation in memory
       or registers.  A pointer to a void type is a generic pointer.  */
    TYPE_CODE_VOID,

    TYPE_CODE_SET,		/* Pascal sets */
    TYPE_CODE_RANGE,		/* Range (integers within spec'd bounds).  */

    /* A string type which is like an array of character but prints
       differently (at least for (the deleted) CHILL).  It does not
       contain a length field as Pascal strings (for many Pascals,
       anyway) do; if we want to deal with such strings, we should use
       a new type code.  */
    TYPE_CODE_STRING,

    /* Unknown type.  The length field is valid if we were able to
       deduce that much about the type, or 0 if we don't even know that.  */
    TYPE_CODE_ERROR,

    /* C++ */
    TYPE_CODE_METHOD,		/* Method type */

    /* Pointer-to-member-function type.  This describes how to access a
       particular member function of a class (possibly a virtual
       member function).  The representation may vary between different
       C++ ABIs.  */
    TYPE_CODE_METHODPTR,

    /* Pointer-to-member type.  This is the offset within a class to some
       particular data member.  The only currently supported representation
       uses an unbiased offset, with -1 representing NULL; this is used
       by the Itanium C++ ABI (used by GCC on all platforms).  */
    TYPE_CODE_MEMBERPTR,

    TYPE_CODE_REF,		/* C++ Reference types */

    TYPE_CODE_CHAR,		/* *real* character type */

    /* Boolean type.  0 is false, 1 is true, and other values are non-boolean
       (e.g. FORTRAN "logical" used as unsigned int).  */
    TYPE_CODE_BOOL,

    /* Fortran */
    TYPE_CODE_COMPLEX,		/* Complex float */

    TYPE_CODE_TYPEDEF,

    TYPE_CODE_NAMESPACE,	/* C++ namespace.  */

    TYPE_CODE_DECFLOAT,		/* Decimal floating point.  */

    TYPE_CODE_MODULE,		/* Fortran module.  */

    /* Internal function type.  */
    TYPE_CODE_INTERNAL_FUNCTION
  };

/* For now allow source to use TYPE_CODE_CLASS for C++ classes, as an
   alias for TYPE_CODE_STRUCT.  This is for DWARF, which has a distinct
   "class" attribute.  Perhaps we should actually have a separate TYPE_CODE
   so that we can print "class" or "struct" depending on what the debug
   info said.  It's not clear we should bother.  */

#define TYPE_CODE_CLASS TYPE_CODE_STRUCT

/* Some constants representing each bit field in the main_type.  See
   the bit-field-specific macros, below, for documentation of each
   constant in this enum.  These enum values are only used with
   init_type.  Note that the values are chosen not to conflict with
   type_instance_flag_value; this lets init_type error-check its
   input.  */

enum type_flag_value
{
  TYPE_FLAG_UNSIGNED = (1 << 19),
  TYPE_FLAG_NOSIGN = (1 << 20),
  TYPE_FLAG_STUB = (1 << 21),
  TYPE_FLAG_TARGET_STUB = (1 << 22),
  TYPE_FLAG_STATIC = (1 << 23),
  TYPE_FLAG_PROTOTYPED = (1 << 24),
  TYPE_FLAG_INCOMPLETE = (1 << 25),
  TYPE_FLAG_VARARGS = (1 << 26),
  TYPE_FLAG_VECTOR = (1 << 27),
  TYPE_FLAG_FIXED_INSTANCE = (1 << 28),
  TYPE_FLAG_STUB_SUPPORTED = (1 << 29),
  TYPE_FLAG_GNU_IFUNC = (1 << 30),

  /* Used for error-checking.  */
  TYPE_FLAG_MIN = TYPE_FLAG_UNSIGNED
};

/* Some bits for the type's instance_flags word.  See the macros below
   for documentation on each bit.  Note that if you add a value here,
   you must update the enum type_flag_value as well.  */
enum type_instance_flag_value
{
  TYPE_INSTANCE_FLAG_CONST = (1 << 0),
  TYPE_INSTANCE_FLAG_VOLATILE = (1 << 1),
  TYPE_INSTANCE_FLAG_CODE_SPACE = (1 << 2),
  TYPE_INSTANCE_FLAG_DATA_SPACE = (1 << 3),
  TYPE_INSTANCE_FLAG_ADDRESS_CLASS_1 = (1 << 4),
  TYPE_INSTANCE_FLAG_ADDRESS_CLASS_2 = (1 << 5),
  TYPE_INSTANCE_FLAG_NOTTEXT = (1 << 6),
  TYPE_INSTANCE_FLAG_RESTRICT = (1 << 7),
  TYPE_INSTANCE_FLAG_IS_CO_SHAPE = (1 << 8),
  /* CUDA - Memory Segments */
  TYPE_INSTANCE_FLAG_CUDA_CODE = (1 << 9),
  TYPE_INSTANCE_FLAG_CUDA_CONST = (1 << 10),
  TYPE_INSTANCE_FLAG_CUDA_GENERIC = (1 << 11),
  TYPE_INSTANCE_FLAG_CUDA_GLOBAL = (1 << 12),
  TYPE_INSTANCE_FLAG_CUDA_PARAM = (1 << 13),
  TYPE_INSTANCE_FLAG_CUDA_SHARED = (1 << 14),
  TYPE_INSTANCE_FLAG_CUDA_TEX = (1 << 15),
  TYPE_INSTANCE_FLAG_CUDA_LOCAL = (1 << 16),
  TYPE_INSTANCE_FLAG_CUDA_REG = (1 << 17),
  TYPE_INSTANCE_FLAG_CUDA_MANAGED = (1<<18),
};

/* Unsigned integer type.  If this is not set for a TYPE_CODE_INT, the
   type is signed (unless TYPE_FLAG_NOSIGN (below) is set).  */

#define TYPE_UNSIGNED(t)	(TYPE_MAIN_TYPE (t)->flag_unsigned)

/* No sign for this type.  In C++, "char", "signed char", and "unsigned
   char" are distinct types; so we need an extra flag to indicate the
   absence of a sign!  */

#define TYPE_NOSIGN(t)		(TYPE_MAIN_TYPE (t)->flag_nosign)

/* This appears in a type's flags word if it is a stub type (e.g., if
   someone referenced a type that wasn't defined in a source file
   via (struct sir_not_appearing_in_this_film *)).  */

#define TYPE_STUB(t)		(TYPE_MAIN_TYPE (t)->flag_stub)

/* The target type of this type is a stub type, and this type needs to
   be updated if it gets un-stubbed in check_typedef.
   Used for arrays and ranges, in which TYPE_LENGTH of the array/range
   gets set based on the TYPE_LENGTH of the target type.
   Also, set for TYPE_CODE_TYPEDEF.  */

#define TYPE_TARGET_STUB(t)	(TYPE_MAIN_TYPE (t)->flag_target_stub)

/* Static type.  If this is set, the corresponding type had 
   a static modifier.
   Note: This may be unnecessary, since static data members
   are indicated by other means (bitpos == -1).  */

#define TYPE_STATIC(t)		(TYPE_MAIN_TYPE (t)->flag_static)

/* This is a function type which appears to have a prototype.  We need
   this for function calls in order to tell us if it's necessary to
   coerce the args, or to just do the standard conversions.  This is
   used with a short field.  */

#define TYPE_PROTOTYPED(t)	(TYPE_MAIN_TYPE (t)->flag_prototyped)

/* This flag is used to indicate that processing for this type
   is incomplete.

   (Mostly intended for HP platforms, where class methods, for
   instance, can be encountered before their classes in the debug
   info; the incomplete type has to be marked so that the class and
   the method can be assigned correct types.)  */

#define TYPE_INCOMPLETE(t)	(TYPE_MAIN_TYPE (t)->flag_incomplete)

/* FIXME drow/2002-06-03:  Only used for methods, but applies as well
   to functions.  */

#define TYPE_VARARGS(t)		(TYPE_MAIN_TYPE (t)->flag_varargs)

/* Identify a vector type.  Gcc is handling this by adding an extra
   attribute to the array type.  We slurp that in as a new flag of a
   type.  This is used only in dwarf2read.c.  */
#define TYPE_VECTOR(t)		(TYPE_MAIN_TYPE (t)->flag_vector)

/* The debugging formats (especially STABS) do not contain enough information
   to represent all Ada types---especially those whose size depends on
   dynamic quantities.  Therefore, the GNAT Ada compiler includes
   extra information in the form of additional type definitions
   connected by naming conventions.  This flag indicates that the 
   type is an ordinary (unencoded) GDB type that has been created from 
   the necessary run-time information, and does not need further 
   interpretation.  Optionally marks ordinary, fixed-size GDB type.  */

#define TYPE_FIXED_INSTANCE(t) (TYPE_MAIN_TYPE (t)->flag_fixed_instance)

/* This debug target supports TYPE_STUB(t).  In the unsupported case we have to
   rely on NFIELDS to be zero etc., see TYPE_IS_OPAQUE ().
   TYPE_STUB(t) with !TYPE_STUB_SUPPORTED(t) may exist if we only guessed
   the TYPE_STUB(t) value (see dwarfread.c).  */

#define TYPE_STUB_SUPPORTED(t)   (TYPE_MAIN_TYPE (t)->flag_stub_supported)

/* Not textual.  By default, GDB treats all single byte integers as
   characters (or elements of strings) unless this flag is set.  */

#define TYPE_NOTTEXT(t)	(TYPE_INSTANCE_FLAGS (t) & TYPE_INSTANCE_FLAG_NOTTEXT)

/* Used only for TYPE_CODE_FUNC where it specifies the real function
   address is returned by this function call.  TYPE_TARGET_TYPE determines the
   final returned function type to be presented to user.  */

#define TYPE_GNU_IFUNC(t)	(TYPE_MAIN_TYPE (t)->flag_gnu_ifunc)

/* Type owner.  If TYPE_OBJFILE_OWNED is true, the type is owned by
   the objfile retrieved as TYPE_OBJFILE.  Otherweise, the type is
   owned by an architecture; TYPE_OBJFILE is NULL in this case.  */

#define TYPE_OBJFILE_OWNED(t) (TYPE_MAIN_TYPE (t)->flag_objfile_owned)
#define TYPE_OWNER(t) TYPE_MAIN_TYPE(t)->owner
#define TYPE_OBJFILE(t) (TYPE_OBJFILE_OWNED(t)? TYPE_OWNER(t).objfile : NULL)

/* True if this type was declared using the "class" keyword.  This is
   only valid for C++ structure types, and only used for displaying
   the type.  If false, the structure was declared as a "struct".  */

#define TYPE_DECLARED_CLASS(t) (TYPE_MAIN_TYPE (t)->flag_declared_class)

/* True if this type is a "flag" enum.  A flag enum is one where all
   the values are pairwise disjoint when "and"ed together.  This
   affects how enum values are printed.  */

#define TYPE_FLAG_ENUM(t) (TYPE_MAIN_TYPE (t)->flag_flag_enum)

/* Constant type.  If this is set, the corresponding type has a
   const modifier.  */

#define TYPE_CONST(t) (TYPE_INSTANCE_FLAGS (t) & TYPE_INSTANCE_FLAG_CONST)

/* Volatile type.  If this is set, the corresponding type has a
   volatile modifier.  */

#define TYPE_VOLATILE(t) \
  (TYPE_INSTANCE_FLAGS (t) & TYPE_INSTANCE_FLAG_VOLATILE)

/* Restrict type.  If this is set, the corresponding type has a
   restrict modifier.  */

#define TYPE_RESTRICT(t) \
  (TYPE_INSTANCE_FLAGS (t) & TYPE_INSTANCE_FLAG_RESTRICT)

/* Instruction-space delimited type.  This is for Harvard architectures
   which have separate instruction and data address spaces (and perhaps
   others).

   GDB usually defines a flat address space that is a superset of the
   architecture's two (or more) address spaces, but this is an extension
   of the architecture's model.

   If TYPE_FLAG_INST is set, an object of the corresponding type
   resides in instruction memory, even if its address (in the extended
   flat address space) does not reflect this.

   Similarly, if TYPE_FLAG_DATA is set, then an object of the 
   corresponding type resides in the data memory space, even if
   this is not indicated by its (flat address space) address.

   If neither flag is set, the default space for functions / methods
   is instruction space, and for data objects is data memory.  */

#define TYPE_CODE_SPACE(t) \
  (TYPE_INSTANCE_FLAGS (t) & TYPE_INSTANCE_FLAG_CODE_SPACE)

#define TYPE_DATA_SPACE(t) \
  (TYPE_INSTANCE_FLAGS (t) & TYPE_INSTANCE_FLAG_DATA_SPACE)

/* CUDA - Memory Segments */
#define TYPE_CUDA_CODE(t) (TYPE_INSTANCE_FLAGS(t) & \
                           TYPE_INSTANCE_FLAG_CUDA_CODE)
#define TYPE_CUDA_CONST(t) (TYPE_INSTANCE_FLAGS(t) & \
                           TYPE_INSTANCE_FLAG_CUDA_CONST)
#define TYPE_CUDA_GENERIC(t) (TYPE_INSTANCE_FLAGS(t) & \
                           (TYPE_INSTANCE_FLAG_CUDA_GENERIC|TYPE_INSTANCE_FLAG_CUDA_MANAGED))
#define TYPE_CUDA_GLOBAL(t) (TYPE_INSTANCE_FLAGS(t) & \
                           TYPE_INSTANCE_FLAG_CUDA_GLOBAL)
#define TYPE_CUDA_PARAM(t) (TYPE_INSTANCE_FLAGS(t) & \
                           TYPE_INSTANCE_FLAG_CUDA_PARAM)
#define TYPE_CUDA_SHARED(t) (TYPE_INSTANCE_FLAGS(t) & \
                           TYPE_INSTANCE_FLAG_CUDA_SHARED)
#define TYPE_CUDA_TEX(t) (TYPE_INSTANCE_FLAGS(t) & \
                           TYPE_INSTANCE_FLAG_CUDA_TEX)
#define TYPE_CUDA_LOCAL(t) (TYPE_INSTANCE_FLAGS(t) & \
                           TYPE_INSTANCE_FLAG_CUDA_LOCAL)
#define TYPE_CUDA_REG(t) (TYPE_INSTANCE_FLAGS(t) & \
                         TYPE_INSTANCE_FLAG_CUDA_REG)
#define TYPE_INSTANCE_FLAG_CUDA_ALL  \
  (TYPE_INSTANCE_FLAG_CUDA_CODE    | \
   TYPE_INSTANCE_FLAG_CUDA_CONST   | \
   TYPE_INSTANCE_FLAG_CUDA_GENERIC | \
   TYPE_INSTANCE_FLAG_CUDA_GLOBAL  | \
   TYPE_INSTANCE_FLAG_CUDA_PARAM   | \
   TYPE_INSTANCE_FLAG_CUDA_SHARED  | \
   TYPE_INSTANCE_FLAG_CUDA_TEX     | \
   TYPE_INSTANCE_FLAG_CUDA_LOCAL   | \
   TYPE_INSTANCE_FLAG_CUDA_MANAGED | \
   TYPE_INSTANCE_FLAG_CUDA_REG)
#define TYPE_CUDA_ALL(t) (TYPE_INSTANCE_FLAGS(t) & \
                          TYPE_INSTANCE_FLAG_CUDA_ALL)

/* Address class flags.  Some environments provide for pointers whose
   size is different from that of a normal pointer or address types
   where the bits are interpreted differently than normal addresses.  The
   TYPE_FLAG_ADDRESS_CLASS_n flags may be used in target specific
   ways to represent these different types of address classes.  */
#define TYPE_ADDRESS_CLASS_1(t) (TYPE_INSTANCE_FLAGS(t) \
                                 & TYPE_INSTANCE_FLAG_ADDRESS_CLASS_1)
#define TYPE_ADDRESS_CLASS_2(t) (TYPE_INSTANCE_FLAGS(t) \
				 & TYPE_INSTANCE_FLAG_ADDRESS_CLASS_2)
#define TYPE_INSTANCE_FLAG_ADDRESS_CLASS_ALL \
  (TYPE_INSTANCE_FLAG_ADDRESS_CLASS_1 | TYPE_INSTANCE_FLAG_ADDRESS_CLASS_2 \
   | TYPE_INSTANCE_FLAG_CUDA_ALL)
#define TYPE_ADDRESS_CLASS_ALL(t) (TYPE_INSTANCE_FLAGS(t) \
				   & TYPE_INSTANCE_FLAG_ADDRESS_CLASS_ALL)

/* Co-Array Fortran.  If this is set, the range type describes a
   coshape.  */

#define TYPE_IS_CO_SHAPE(t)	(TYPE_INSTANCE_FLAGS (t) & TYPE_INSTANCE_FLAG_IS_CO_SHAPE)

/* Determine which field of the union main_type.fields[x].loc is used.  */

enum field_loc_kind
  {
    FIELD_LOC_KIND_BITPOS,	/* bitpos */
    FIELD_LOC_KIND_ENUMVAL,	/* enumval */
    FIELD_LOC_KIND_PHYSADDR,	/* physaddr */
    FIELD_LOC_KIND_PHYSNAME,	/* physname */
    FIELD_LOC_KIND_DWARF_BLOCK	/* dwarf_block */
  };

/* A discriminant to determine which field in the main_type.type_specific
   union is being used, if any.

   For types such as TYPE_CODE_FLT or TYPE_CODE_FUNC, the use of this
   discriminant is really redundant, as we know from the type code
   which field is going to be used.  As such, it would be possible to
   reduce the size of this enum in order to save a bit or two for
   other fields of struct main_type.  But, since we still have extra
   room , and for the sake of clarity and consistency, we treat all fields
   of the union the same way.  */

enum type_specific_kind
{
  TYPE_SPECIFIC_NONE,
  TYPE_SPECIFIC_CPLUS_STUFF,
  TYPE_SPECIFIC_GNAT_STUFF,
  TYPE_SPECIFIC_FLOATFORMAT,
  TYPE_SPECIFIC_FUNC
};

/* This structure is space-critical.
   Its layout has been tweaked to reduce the space used.  */

struct main_type
{
  /* Code for kind of type.  */

  ENUM_BITFIELD(type_code) code : 8;

  /* Flags about this type.  These fields appear at this location
     because they packs nicely here.  See the TYPE_* macros for
     documentation about these fields.  */

  unsigned int flag_unsigned : 1;
  unsigned int flag_nosign : 1;
  unsigned int flag_stub : 1;
  unsigned int flag_target_stub : 1;
  unsigned int flag_static : 1;
  unsigned int flag_prototyped : 1;
  unsigned int flag_incomplete : 1;
  unsigned int flag_varargs : 1;
  unsigned int flag_vector : 1;
  unsigned int flag_stub_supported : 1;
  unsigned int flag_gnu_ifunc : 1;
  unsigned int flag_fixed_instance : 1;
  unsigned int flag_objfile_owned : 1;
  /* True if this type was declared with "class" rather than
     "struct".  */
  unsigned int flag_declared_class : 1;

  /* True if this is an enum type with disjoint values.  This affects
     how the enum is printed.  */

  unsigned int flag_flag_enum : 1;

  /* A discriminant telling us which field of the type_specific union
     is being used for this type, if any.  */
  ENUM_BITFIELD(type_specific_kind) type_specific_field : 3;

  /* Number of fields described for this type.  This field appears at
     this location because it packs nicely here.  */

  short nfields;

  /* Field number of the virtual function table pointer in
     VPTR_BASETYPE.  If -1, we were unable to find the virtual
     function table pointer in initial symbol reading, and
     get_vptr_fieldno should be called to find it if possible.
     get_vptr_fieldno will update this field if possible.
     Otherwise the value is left at -1.

     Unused if this type does not have virtual functions.

     This field appears at this location because it packs nicely here.  */

  short vptr_fieldno;

  /* Name of this type, or NULL if none.

     This is used for printing only, except by poorly designed C++ code.
     For looking up a name, look for a symbol in the VAR_DOMAIN.
     This is generally allocated in the objfile's obstack.
     However coffread.c uses malloc.  */

  const char *name;

  /* Tag name for this type, or NULL if none.  This means that the
     name of the type consists of a keyword followed by the tag name.
     Which keyword is determined by the type code ("struct" for
     TYPE_CODE_STRUCT, etc.).  As far as I know C/C++ are the only languages
     with this feature.

     This is used for printing only, except by poorly designed C++ code.
     For looking up a name, look for a symbol in the STRUCT_DOMAIN.
     One more legitimate use is that if TYPE_FLAG_STUB is set, this is
     the name to use to look for definitions in other files.  */

  const char *tag_name;

  /* Every type is now associated with a particular objfile, and the
     type is allocated on the objfile_obstack for that objfile.  One problem
     however, is that there are times when gdb allocates new types while
     it is not in the process of reading symbols from a particular objfile.
     Fortunately, these happen when the type being created is a derived
     type of an existing type, such as in lookup_pointer_type().  So
     we can just allocate the new type using the same objfile as the
     existing type, but to do this we need a backpointer to the objfile
     from the existing type.  Yes this is somewhat ugly, but without
     major overhaul of the internal type system, it can't be avoided
     for now.  */

  union type_owner
    {
      struct objfile *objfile;
      struct gdbarch *gdbarch;
    } owner;

  /* For a pointer type, describes the type of object pointed to.
     For an array type, describes the type of the elements.
     For a function or method type, describes the type of the return value.
     For a range type, describes the type of the full range.
     For a complex type, describes the type of each coordinate.
     For a special record or union type encoding a dynamic-sized type
     in GNAT, a memoized pointer to a corresponding static version of
     the type.
     Unused otherwise.  */

  struct type *target_type;

  /* For structure and union types, a description of each field.
     For set and pascal array types, there is one "field",
     whose type is the domain type of the set or array.
     For range types, there are two "fields",
     the minimum and maximum values (both inclusive).
     For enum types, each possible value is described by one "field".
     For a function or method type, a "field" for each parameter.
     For C++ classes, there is one field for each base class (if it is
     a derived class) plus one field for each class data member.  Member
     functions are recorded elsewhere.

     Using a pointer to a separate array of fields
     allows all types to have the same size, which is useful
     because we can allocate the space for a type before
     we know what to put in it.  */

  union 
  {
    struct field
    {
      union field_location
      {
	/* Position of this field, counting in bits from start of
	   containing structure.  For gdbarch_bits_big_endian=1
	   targets, it is the bit offset to the MSB.  For
	   gdbarch_bits_big_endian=0 targets, it is the bit offset to
	   the LSB.  */

	int bitpos;

	/* Enum value.  */
	LONGEST enumval;

	/* For a static field, if TYPE_FIELD_STATIC_HAS_ADDR then physaddr
	   is the location (in the target) of the static field.
	   Otherwise, physname is the mangled label of the static field.  */

	CORE_ADDR physaddr;
	const char *physname;

	/* The field location can be computed by evaluating the following DWARF
	   block.  Its DATA is allocated on objfile_obstack - no CU load is
	   needed to access it.  */

	struct dwarf2_locexpr_baton *dwarf_block;
      }
      loc;

      /* For a function or member type, this is 1 if the argument is marked
	 artificial.  Artificial arguments should not be shown to the
	 user.  For TYPE_CODE_RANGE it is set if the specific bound is not
	 defined.  */
      unsigned int artificial : 1;

      /* Discriminant for union field_location.  */
      ENUM_BITFIELD(field_loc_kind) loc_kind : 3;

      /* Size of this field, in bits, or zero if not packed.
	 If non-zero in an array type, indicates the element size in
	 bits (used only in Ada at the moment).
	 For an unpacked field, the field's type's length
	 says how many bytes the field occupies.  */

      unsigned int bitsize : 28;

      /* In a struct or union type, type of this field.
	 In a function or member type, type of this argument.
	 In an array type, the domain-type of the array.  */

      struct type *type;

      /* Name of field, value or argument.
	 NULL for range bounds, array domains, and member function
	 arguments.  */

      const char *name;
    } *fields;

    /* Union member used for range types.  */

    struct range_bounds
    {
      /* Low bound of range.  */

      LONGEST low;

      /* High bound of range.  */

      LONGEST high;

      /* Flags indicating whether the values of low and high are
         valid.  When true, the respective range value is
         undefined.  Currently used only for FORTRAN arrays.  */
           
      char low_undefined;
      char high_undefined;

      void* low_baton;
      void* high_baton;
      void* count_baton;
      void* baton_function;

      void* lstride_baton;
      void* stride_baton;
      void* soffset_baton;

      LONGEST lstride_value;
      LONGEST stride_value;
      LONGEST soffset_value;
    } *bounds;

  } flds_bnds;

  /* For types with virtual functions (TYPE_CODE_STRUCT), VPTR_BASETYPE
     is the base class which defined the virtual function table pointer.  

     For types that are pointer to member types (TYPE_CODE_METHODPTR,
     TYPE_CODE_MEMBERPTR), VPTR_BASETYPE is the type that this pointer
     is a member of.

     For method types (TYPE_CODE_METHOD), VPTR_BASETYPE is the aggregate
     type that contains the method.

     Unused otherwise.  */

  struct type *vptr_basetype;

  /* Slot to point to additional language-specific fields of this type.  */

  union type_specific
  {
    /* CPLUS_STUFF is for TYPE_CODE_STRUCT.  It is initialized to point to
       cplus_struct_default, a default static instance of a struct
       cplus_struct_type.  */

    struct cplus_struct_type *cplus_stuff;

    /* GNAT_STUFF is for types for which the GNAT Ada compiler
       provides additional information.  */
    struct gnat_aux_type *gnat_stuff;

    /* FLOATFORMAT is for TYPE_CODE_FLT.  It is a pointer to two
       floatformat objects that describe the floating-point value
       that resides within the type.  The first is for big endian
       targets and the second is for little endian targets.  */

    const struct floatformat **floatformat;

    /* For TYPE_CODE_FUNC types,  */
    struct func_type *func_stuff;
  } type_specific;
};

/* A ``struct type'' describes a particular instance of a type, with
   some particular qualification.  */
struct type
{
  /* Type that is a pointer to this type.
     NULL if no such pointer-to type is known yet.
     The debugger may add the address of such a type
     if it has to construct one later.  */

  struct type *pointer_type;

  /* C++: also need a reference type.  */

  struct type *reference_type;

  /* Variant chain.  This points to a type that differs from this one only
     in qualifiers and length.  Currently, the possible qualifiers are
     const, volatile, code-space, data-space, and address class.  The
     length may differ only when one of the address class flags are set.
     The variants are linked in a circular ring and share MAIN_TYPE.  */
  struct type *chain;

  /* Flags specific to this instance of the type, indicating where
     on the ring we are.

     For TYPE_CODE_TYPEDEF the flags of the typedef type should be binary
     or-ed with the target type, with a special case for address class and
     space class.  For example if this typedef does not specify any new
     qualifiers, TYPE_INSTANCE_FLAGS is 0 and the instance flags are
     completely inherited from the target type.  No qualifiers can be cleared
     by the typedef.  See also check_typedef.  */
  int instance_flags;

  /* Length of storage for a value of this type.  This is what
     sizeof(type) would return; use it for address arithmetic,
     memory reads and writes, etc.  This size includes padding.  For
     example, an i386 extended-precision floating point value really
     only occupies ten bytes, but most ABI's declare its size to be
     12 bytes, to preserve alignment.  A `struct type' representing
     such a floating-point type would have a `length' value of 12,
     even though the last two bytes are unused.

     There's a bit of a host/target mess here, if you're concerned
     about machines whose bytes aren't eight bits long, or who don't
     have byte-addressed memory.  Various places pass this to memcpy
     and such, meaning it must be in units of host bytes.  Various
     other places expect they can calculate addresses by adding it
     and such, meaning it must be in units of target bytes.  For
     some DSP targets, in which HOST_CHAR_BIT will (presumably) be 8
     and TARGET_CHAR_BIT will be (say) 32, this is a problem.

     One fix would be to make this field in bits (requiring that it
     always be a multiple of HOST_CHAR_BIT and TARGET_CHAR_BIT) ---
     the other choice would be to make it consistently in units of
     HOST_CHAR_BIT.  However, this would still fail to address
     machines based on a ternary or decimal representation.  */
  
  unsigned length;

  /* Core type, shared by a group of qualified types.  */
  struct main_type *main_type;
};

#define	NULL_TYPE ((struct type *) 0)

/* C++ language-specific information for TYPE_CODE_STRUCT and TYPE_CODE_UNION
   nodes.  */

struct cplus_struct_type
  {
    /* Number of base classes this type derives from.  The baseclasses are
       stored in the first N_BASECLASSES fields (i.e. the `fields' field of
       the struct type).  I think only the `type' field of such a field has
       any meaning.  */

    short n_baseclasses;

    /* Number of methods with unique names.  All overloaded methods with
       the same name count only once.  */

    short nfn_fields;

    /* Number of template arguments.  */
    unsigned short n_template_arguments;

    /* One if this struct is a dynamic class, as defined by the
       Itanium C++ ABI: if it requires a virtual table pointer,
       because it or any of its base classes have one or more virtual
       member functions or virtual base classes.  Minus one if not
       dynamic.  Zero if not yet computed.  */
    int is_dynamic : 2;

    /* Non-zero if this type came from a Java CU.  */
    unsigned int is_java : 1;

    /* For derived classes, the number of base classes is given by
       n_baseclasses and virtual_field_bits is a bit vector containing
       one bit per base class.  If the base class is virtual, the
       corresponding bit will be set.
       I.E, given:

       class A{};
       class B{};
       class C : public B, public virtual A {};

       B is a baseclass of C; A is a virtual baseclass for C.
       This is a C++ 2.0 language feature.  */

    B_TYPE *virtual_field_bits;

    /* For classes with private fields, the number of fields is given by
       nfields and private_field_bits is a bit vector containing one bit
       per field.
       If the field is private, the corresponding bit will be set.  */

    B_TYPE *private_field_bits;

    /* For classes with protected fields, the number of fields is given by
       nfields and protected_field_bits is a bit vector containing one bit
       per field.
       If the field is private, the corresponding bit will be set.  */

    B_TYPE *protected_field_bits;

    /* For classes with fields to be ignored, either this is optimized out
       or this field has length 0.  */

    B_TYPE *ignore_field_bits;

    /* For classes, structures, and unions, a description of each field,
       which consists of an overloaded name, followed by the types of
       arguments that the method expects, and then the name after it
       has been renamed to make it distinct.

       fn_fieldlists points to an array of nfn_fields of these.  */

    struct fn_fieldlist
      {

	/* The overloaded name.
	   This is generally allocated in the objfile's obstack.
	   However stabsread.c sometimes uses malloc.  */

	const char *name;

	/* The number of methods with this name.  */

	int length;

	/* The list of methods.  */

	struct fn_field
	  {

	    /* If is_stub is clear, this is the mangled name which we can
	       look up to find the address of the method (FIXME: it would
	       be cleaner to have a pointer to the struct symbol here
	       instead).  */

	    /* If is_stub is set, this is the portion of the mangled
	       name which specifies the arguments.  For example, "ii",
	       if there are two int arguments, or "" if there are no
	       arguments.  See gdb_mangle_name for the conversion from this
	       format to the one used if is_stub is clear.  */

	    const char *physname;

	    /* The function type for the method.
	       (This comment used to say "The return value of the method",
	       but that's wrong.  The function type 
	       is expected here, i.e. something with TYPE_CODE_FUNC,
	       and *not* the return-value type).  */

	    struct type *type;

	    /* For virtual functions.
	       First baseclass that defines this virtual function.  */

	    struct type *fcontext;

	    /* Attributes.  */

	    unsigned int is_const:1;
	    unsigned int is_volatile:1;
	    unsigned int is_private:1;
	    unsigned int is_protected:1;
	    unsigned int is_public:1;
	    unsigned int is_abstract:1;
	    unsigned int is_static:1;
	    unsigned int is_final:1;
	    unsigned int is_synchronized:1;
	    unsigned int is_native:1;
	    unsigned int is_artificial:1;

	    /* A stub method only has some fields valid (but they are enough
	       to reconstruct the rest of the fields).  */
	    unsigned int is_stub:1;

	    /* True if this function is a constructor, false
	       otherwise.  */
	    unsigned int is_constructor : 1;

	    /* Unused.  */
	    unsigned int dummy:3;

	    /* Index into that baseclass's virtual function table,
	       minus 2; else if static: VOFFSET_STATIC; else: 0.  */

	    unsigned int voffset:16;

#define VOFFSET_STATIC 1

	  }
	 *fn_fields;

      }
     *fn_fieldlists;

    /* typedefs defined inside this class.  TYPEDEF_FIELD points to an array of
       TYPEDEF_FIELD_COUNT elements.  */
    struct typedef_field
      {
	/* Unqualified name to be prefixed by owning class qualified name.  */
	const char *name;

	/* Type this typedef named NAME represents.  */
	struct type *type;
      }
    *typedef_field;
    unsigned typedef_field_count;

    /* The template arguments.  This is an array with
       N_TEMPLATE_ARGUMENTS elements.  This is NULL for non-template
       classes.  */
    struct symbol **template_arguments;
  };

/* Struct used to store conversion rankings.  */
struct rank
  {
    short rank;

    /* When two conversions are of the same type and therefore have the same
       rank, subrank is used to differentiate the two.
       Eg: Two derived-class-pointer to base-class-pointer conversions would
       both have base pointer conversion rank, but the conversion with the
       shorter distance to the ancestor is preferable.  'subrank' would be used
       to reflect that.  */
    short subrank;
  };

/* Struct used for ranking a function for overload resolution.  */
struct badness_vector
  {
    int length;
    struct rank *rank;
  };

/* GNAT Ada-specific information for various Ada types.  */
struct gnat_aux_type
  {
    /* Parallel type used to encode information about dynamic types
       used in Ada (such as variant records, variable-size array,
       etc).  */
    struct type* descriptive_type;
  };

/* For TYPE_CODE_FUNC types,  */
struct func_type
  {
    /* The calling convention for targets supporting multiple ABIs.  Right now
       this is only fetched from the Dwarf-2 DW_AT_calling_convention
       attribute.  */
    unsigned calling_convention;

    /* Only those DW_TAG_GNU_call_site's in this function that have
       DW_AT_GNU_tail_call set are linked in this list.  Function without its
       tail call list complete (DW_AT_GNU_all_tail_call_sites or its superset
       DW_AT_GNU_all_call_sites) has TAIL_CALL_LIST NULL, even if some
       DW_TAG_GNU_call_site's exist in such function. */
    struct call_site *tail_call_list;
  };

/* struct call_site_parameter can be referenced in callees by several ways.  */

enum call_site_parameter_kind
{
  /* Use field call_site_parameter.u.dwarf_reg.  */
  CALL_SITE_PARAMETER_DWARF_REG,

  /* Use field call_site_parameter.u.fb_offset.  */
  CALL_SITE_PARAMETER_FB_OFFSET,

  /* Use field call_site_parameter.u.param_offset.  */
  CALL_SITE_PARAMETER_PARAM_OFFSET
};

/* A place where a function gets called from, represented by
   DW_TAG_GNU_call_site.  It can be looked up from symtab->call_site_htab.  */

struct call_site
  {
    /* Address of the first instruction after this call.  It must be the first
       field as we overload core_addr_hash and core_addr_eq for it.  */
    CORE_ADDR pc;

    /* List successor with head in FUNC_TYPE.TAIL_CALL_LIST.  */
    struct call_site *tail_call_next;

    /* Describe DW_AT_GNU_call_site_target.  Missing attribute uses
       FIELD_LOC_KIND_DWARF_BLOCK with FIELD_DWARF_BLOCK == NULL.  */
    struct
      {
	union field_location loc;

	/* Discriminant for union field_location.  */
	ENUM_BITFIELD(field_loc_kind) loc_kind : 3;
      }
    target;

    /* Size of the PARAMETER array.  */
    unsigned parameter_count;

    /* CU of the function where the call is located.  It gets used for DWARF
       blocks execution in the parameter array below.  */
    struct dwarf2_per_cu_data *per_cu;

    /* Describe DW_TAG_GNU_call_site's DW_TAG_formal_parameter.  */
    struct call_site_parameter
      {
	ENUM_BITFIELD (call_site_parameter_kind) kind : 2;

	union call_site_parameter_u
	  {
	    /* DW_TAG_formal_parameter's DW_AT_location's DW_OP_regX as DWARF
	       register number, for register passed parameters.  */
	    int dwarf_reg;

	    /* Offset from the callee's frame base, for stack passed parameters.
	       This equals offset from the caller's stack pointer.  */
	    CORE_ADDR fb_offset;

	    /* Offset relative to the start of this PER_CU to
	       DW_TAG_formal_parameter which is referenced by both caller and
	       the callee.  */
	    cu_offset param_offset;
	  }
	u;

	/* DW_TAG_formal_parameter's DW_AT_GNU_call_site_value.  It is never
	   NULL.  */
	const gdb_byte *value;
	size_t value_size;

	/* DW_TAG_formal_parameter's DW_AT_GNU_call_site_data_value.  It may be
	   NULL if not provided by DWARF.  */
	const gdb_byte *data_value;
	size_t data_value_size;
      }
    parameter[1];
  };

/* The default value of TYPE_CPLUS_SPECIFIC(T) points to the
   this shared static structure.  */

extern const struct cplus_struct_type cplus_struct_default;

extern void allocate_cplus_struct_type (struct type *);

#define INIT_CPLUS_SPECIFIC(type) \
  (TYPE_SPECIFIC_FIELD (type) = TYPE_SPECIFIC_CPLUS_STUFF, \
   TYPE_RAW_CPLUS_SPECIFIC (type) = (struct cplus_struct_type*) \
   &cplus_struct_default)

#define ALLOCATE_CPLUS_STRUCT_TYPE(type) allocate_cplus_struct_type (type)

#define HAVE_CPLUS_STRUCT(type) \
  (TYPE_SPECIFIC_FIELD (type) == TYPE_SPECIFIC_CPLUS_STUFF \
   && TYPE_RAW_CPLUS_SPECIFIC (type) !=  &cplus_struct_default)

extern const struct gnat_aux_type gnat_aux_default;

extern void allocate_gnat_aux_type (struct type *);

#define INIT_GNAT_SPECIFIC(type) \
  (TYPE_SPECIFIC_FIELD (type) = TYPE_SPECIFIC_GNAT_STUFF, \
   TYPE_GNAT_SPECIFIC (type) = (struct gnat_aux_type *) &gnat_aux_default)
#define ALLOCATE_GNAT_AUX_TYPE(type) allocate_gnat_aux_type (type)
/* A macro that returns non-zero if the type-specific data should be
   read as "gnat-stuff".  */
#define HAVE_GNAT_AUX_INFO(type) \
  (TYPE_SPECIFIC_FIELD (type) == TYPE_SPECIFIC_GNAT_STUFF)

#define INIT_FUNC_SPECIFIC(type)					       \
  (TYPE_SPECIFIC_FIELD (type) = TYPE_SPECIFIC_FUNC,			       \
   TYPE_MAIN_TYPE (type)->type_specific.func_stuff			       \
     = TYPE_ZALLOC (type,						       \
		    sizeof (*TYPE_MAIN_TYPE (type)->type_specific.func_stuff)))

#define TYPE_INSTANCE_FLAGS(thistype) (thistype)->instance_flags
#define TYPE_MAIN_TYPE(thistype) (thistype)->main_type
#define TYPE_NAME(thistype) TYPE_MAIN_TYPE(thistype)->name
#define TYPE_TAG_NAME(type) TYPE_MAIN_TYPE(type)->tag_name
/* CUDA - find the target type with the same instance flags */
#define FIND_TARGET_TYPE(thistype) (find_target_type_with_instance_flags(thistype))
#define TYPE_TARGET_TYPE(thistype) TYPE_MAIN_TYPE(thistype)->target_type
#define TYPE_POINTER_TYPE(thistype) (thistype)->pointer_type
#define TYPE_REFERENCE_TYPE(thistype) (thistype)->reference_type
#define TYPE_CHAIN(thistype) (thistype)->chain
/* Note that if thistype is a TYPEDEF type, you have to call check_typedef.
   But check_typedef does set the TYPE_LENGTH of the TYPEDEF type,
   so you only have to call check_typedef once.  Since allocate_value
   calls check_typedef, TYPE_LENGTH (VALUE_TYPE (X)) is safe.  */
#define TYPE_LENGTH(thistype) (thistype)->length
/* Note that TYPE_CODE can be TYPE_CODE_TYPEDEF, so if you want the real
   type, you need to do TYPE_CODE (check_type (this_type)).  */
#define TYPE_CODE(thistype) TYPE_MAIN_TYPE(thistype)->code
#define TYPE_NFIELDS(thistype) TYPE_MAIN_TYPE(thistype)->nfields
#define TYPE_FIELDS(thistype) TYPE_MAIN_TYPE(thistype)->flds_bnds.fields

#define TYPE_INDEX_TYPE(type) TYPE_FIELD_TYPE (type, 0)
#define TYPE_RANGE_DATA(thistype) TYPE_MAIN_TYPE(thistype)->flds_bnds.bounds
#define TYPE_LOW_BOUND(range_type) TYPE_RANGE_DATA(range_type)->low
#define TYPE_HIGH_BOUND(range_type) TYPE_RANGE_DATA(range_type)->high
#define TYPE_LOW_BOUND_BATON(range_type) TYPE_RANGE_DATA(range_type)->low_baton
#define TYPE_HIGH_BOUND_BATON(range_type) TYPE_RANGE_DATA(range_type)->high_baton
#define TYPE_COUNT_BOUND_BATON(range_type) TYPE_RANGE_DATA(range_type)->count_baton
#define TYPE_LSTRIDE_BATON(range_type) TYPE_RANGE_DATA(range_type)->lstride_baton
#define TYPE_LSTRIDE_VALUE(range_type) TYPE_RANGE_DATA(range_type)->lstride_value
#define TYPE_STRIDE_BATON(range_type) TYPE_RANGE_DATA(range_type)->stride_baton
#define TYPE_STRIDE_VALUE(range_type) TYPE_RANGE_DATA(range_type)->stride_value
#define TYPE_SOFFSET_BATON(range_type) TYPE_RANGE_DATA(range_type)->soffset_baton
#define TYPE_SOFFSET_VALUE(range_type) TYPE_RANGE_DATA(range_type)->soffset_value
#define TYPE_BOUND_BATON_FUNCTION(range_type) TYPE_RANGE_DATA(range_type)->baton_function

#define TYPE_LOW_BOUND_UNDEFINED(range_type) \
   TYPE_RANGE_DATA(range_type)->low_undefined
#define TYPE_HIGH_BOUND_UNDEFINED(range_type) \
   TYPE_RANGE_DATA(range_type)->high_undefined

/* Moto-specific stuff for FORTRAN arrays.  */

#define TYPE_ARRAY_UPPER_BOUND_IS_UNDEFINED(arraytype) \
   TYPE_HIGH_BOUND_UNDEFINED(TYPE_INDEX_TYPE(arraytype))
#define TYPE_ARRAY_LOWER_BOUND_IS_UNDEFINED(arraytype) \
   TYPE_LOW_BOUND_UNDEFINED(TYPE_INDEX_TYPE(arraytype))

#define TYPE_ARRAY_UPPER_BOUND_VALUE(arraytype) \
   (TYPE_HIGH_BOUND(TYPE_INDEX_TYPE((arraytype))))

#define TYPE_ARRAY_LOWER_BOUND_VALUE(arraytype) \
   (TYPE_LOW_BOUND(TYPE_INDEX_TYPE((arraytype))))

/* C++ */

#define TYPE_VPTR_BASETYPE(thistype) TYPE_MAIN_TYPE(thistype)->vptr_basetype
#define TYPE_DOMAIN_TYPE(thistype) TYPE_MAIN_TYPE(thistype)->vptr_basetype
#define TYPE_VPTR_FIELDNO(thistype) TYPE_MAIN_TYPE(thistype)->vptr_fieldno
#define TYPE_FN_FIELDS(thistype) TYPE_CPLUS_SPECIFIC(thistype)->fn_fields
#define TYPE_NFN_FIELDS(thistype) TYPE_CPLUS_SPECIFIC(thistype)->nfn_fields
#define TYPE_SPECIFIC_FIELD(thistype) \
  TYPE_MAIN_TYPE(thistype)->type_specific_field
#define	TYPE_TYPE_SPECIFIC(thistype) TYPE_MAIN_TYPE(thistype)->type_specific
/* We need this tap-dance with the TYPE_RAW_SPECIFIC because of the case
   where we're trying to print an Ada array using the C language.
   In that case, there is no "cplus_stuff", but the C language assumes
   that there is.  What we do, in that case, is pretend that there is
   an implicit one which is the default cplus stuff.  */
#define TYPE_CPLUS_SPECIFIC(thistype) \
   (!HAVE_CPLUS_STRUCT(thistype) \
    ? (struct cplus_struct_type*)&cplus_struct_default \
    : TYPE_RAW_CPLUS_SPECIFIC(thistype))
#define TYPE_RAW_CPLUS_SPECIFIC(thistype) TYPE_MAIN_TYPE(thistype)->type_specific.cplus_stuff
#define TYPE_FLOATFORMAT(thistype) TYPE_MAIN_TYPE(thistype)->type_specific.floatformat
#define TYPE_GNAT_SPECIFIC(thistype) TYPE_MAIN_TYPE(thistype)->type_specific.gnat_stuff
#define TYPE_DESCRIPTIVE_TYPE(thistype) TYPE_GNAT_SPECIFIC(thistype)->descriptive_type
#define TYPE_CALLING_CONVENTION(thistype) TYPE_MAIN_TYPE(thistype)->type_specific.func_stuff->calling_convention
#define TYPE_TAIL_CALL_LIST(thistype) TYPE_MAIN_TYPE(thistype)->type_specific.func_stuff->tail_call_list
#define TYPE_BASECLASS(thistype,index) TYPE_FIELD_TYPE(thistype, index)
#define TYPE_N_BASECLASSES(thistype) TYPE_CPLUS_SPECIFIC(thistype)->n_baseclasses
#define TYPE_BASECLASS_NAME(thistype,index) TYPE_FIELD_NAME(thistype, index)
#define TYPE_BASECLASS_BITPOS(thistype,index) TYPE_FIELD_BITPOS(thistype,index)
#define BASETYPE_VIA_PUBLIC(thistype, index) \
  ((!TYPE_FIELD_PRIVATE(thistype, index)) && (!TYPE_FIELD_PROTECTED(thistype, index)))
#define TYPE_CPLUS_DYNAMIC(thistype) TYPE_CPLUS_SPECIFIC (thistype)->is_dynamic
#define TYPE_CPLUS_REALLY_JAVA(thistype) TYPE_CPLUS_SPECIFIC (thistype)->is_java

#define BASETYPE_VIA_VIRTUAL(thistype, index) \
  (TYPE_CPLUS_SPECIFIC(thistype)->virtual_field_bits == NULL ? 0 \
    : B_TST(TYPE_CPLUS_SPECIFIC(thistype)->virtual_field_bits, (index)))

#define FIELD_TYPE(thisfld) ((thisfld).type)
#define FIELD_NAME(thisfld) ((thisfld).name)
#define FIELD_LOC_KIND(thisfld) ((thisfld).loc_kind)
#define FIELD_BITPOS_LVAL(thisfld) ((thisfld).loc.bitpos)
#define FIELD_BITPOS(thisfld) (FIELD_BITPOS_LVAL (thisfld) + 0)
#define FIELD_ENUMVAL_LVAL(thisfld) ((thisfld).loc.enumval)
#define FIELD_ENUMVAL(thisfld) (FIELD_ENUMVAL_LVAL (thisfld) + 0)
#define FIELD_STATIC_PHYSNAME(thisfld) ((thisfld).loc.physname)
#define FIELD_STATIC_PHYSADDR(thisfld) ((thisfld).loc.physaddr)
#define FIELD_DWARF_BLOCK(thisfld) ((thisfld).loc.dwarf_block)
#define SET_FIELD_BITPOS(thisfld, bitpos)			\
  (FIELD_LOC_KIND (thisfld) = FIELD_LOC_KIND_BITPOS,		\
   FIELD_BITPOS_LVAL (thisfld) = (bitpos))
#define SET_FIELD_ENUMVAL(thisfld, enumval)			\
  (FIELD_LOC_KIND (thisfld) = FIELD_LOC_KIND_ENUMVAL,		\
   FIELD_ENUMVAL_LVAL (thisfld) = (enumval))
#define SET_FIELD_PHYSNAME(thisfld, name)			\
  (FIELD_LOC_KIND (thisfld) = FIELD_LOC_KIND_PHYSNAME,		\
   FIELD_STATIC_PHYSNAME (thisfld) = (name))
#define SET_FIELD_PHYSADDR(thisfld, addr)			\
  (FIELD_LOC_KIND (thisfld) = FIELD_LOC_KIND_PHYSADDR,		\
   FIELD_STATIC_PHYSADDR (thisfld) = (addr))
#define SET_FIELD_DWARF_BLOCK(thisfld, addr)			\
  (FIELD_LOC_KIND (thisfld) = FIELD_LOC_KIND_DWARF_BLOCK,	\
   FIELD_DWARF_BLOCK (thisfld) = (addr))
#define FIELD_ARTIFICIAL(thisfld) ((thisfld).artificial)
#define FIELD_BITSIZE(thisfld) ((thisfld).bitsize)

#define TYPE_FIELD(thistype, n) TYPE_MAIN_TYPE(thistype)->flds_bnds.fields[n]
#define TYPE_FIELD_TYPE(thistype, n) FIELD_TYPE(TYPE_FIELD(thistype, n))
#define TYPE_FIELD_NAME(thistype, n) FIELD_NAME(TYPE_FIELD(thistype, n))
#define TYPE_FIELD_LOC_KIND(thistype, n) FIELD_LOC_KIND (TYPE_FIELD (thistype, n))
#define TYPE_FIELD_BITPOS(thistype, n) FIELD_BITPOS (TYPE_FIELD (thistype, n))
#define TYPE_FIELD_ENUMVAL(thistype, n) FIELD_ENUMVAL (TYPE_FIELD (thistype, n))
#define TYPE_FIELD_STATIC_PHYSNAME(thistype, n) FIELD_STATIC_PHYSNAME (TYPE_FIELD (thistype, n))
#define TYPE_FIELD_STATIC_PHYSADDR(thistype, n) FIELD_STATIC_PHYSADDR (TYPE_FIELD (thistype, n))
#define TYPE_FIELD_DWARF_BLOCK(thistype, n) FIELD_DWARF_BLOCK (TYPE_FIELD (thistype, n))
#define TYPE_FIELD_ARTIFICIAL(thistype, n) FIELD_ARTIFICIAL(TYPE_FIELD(thistype,n))
#define TYPE_FIELD_BITSIZE(thistype, n) FIELD_BITSIZE(TYPE_FIELD(thistype,n))
#define TYPE_FIELD_PACKED(thistype, n) (FIELD_BITSIZE(TYPE_FIELD(thistype,n))!=0)

#define TYPE_FIELD_PRIVATE_BITS(thistype) \
  TYPE_CPLUS_SPECIFIC(thistype)->private_field_bits
#define TYPE_FIELD_PROTECTED_BITS(thistype) \
  TYPE_CPLUS_SPECIFIC(thistype)->protected_field_bits
#define TYPE_FIELD_IGNORE_BITS(thistype) \
  TYPE_CPLUS_SPECIFIC(thistype)->ignore_field_bits
#define TYPE_FIELD_VIRTUAL_BITS(thistype) \
  TYPE_CPLUS_SPECIFIC(thistype)->virtual_field_bits
#define SET_TYPE_FIELD_PRIVATE(thistype, n) \
  B_SET (TYPE_CPLUS_SPECIFIC(thistype)->private_field_bits, (n))
#define SET_TYPE_FIELD_PROTECTED(thistype, n) \
  B_SET (TYPE_CPLUS_SPECIFIC(thistype)->protected_field_bits, (n))
#define SET_TYPE_FIELD_IGNORE(thistype, n) \
  B_SET (TYPE_CPLUS_SPECIFIC(thistype)->ignore_field_bits, (n))
#define SET_TYPE_FIELD_VIRTUAL(thistype, n) \
  B_SET (TYPE_CPLUS_SPECIFIC(thistype)->virtual_field_bits, (n))
#define TYPE_FIELD_PRIVATE(thistype, n) \
  (TYPE_CPLUS_SPECIFIC(thistype)->private_field_bits == NULL ? 0 \
    : B_TST(TYPE_CPLUS_SPECIFIC(thistype)->private_field_bits, (n)))
#define TYPE_FIELD_PROTECTED(thistype, n) \
  (TYPE_CPLUS_SPECIFIC(thistype)->protected_field_bits == NULL ? 0 \
    : B_TST(TYPE_CPLUS_SPECIFIC(thistype)->protected_field_bits, (n)))
#define TYPE_FIELD_IGNORE(thistype, n) \
  (TYPE_CPLUS_SPECIFIC(thistype)->ignore_field_bits == NULL ? 0 \
    : B_TST(TYPE_CPLUS_SPECIFIC(thistype)->ignore_field_bits, (n)))
#define TYPE_FIELD_VIRTUAL(thistype, n) \
  (TYPE_CPLUS_SPECIFIC(thistype)->virtual_field_bits == NULL ? 0 \
    : B_TST(TYPE_CPLUS_SPECIFIC(thistype)->virtual_field_bits, (n)))

#define TYPE_FN_FIELDLISTS(thistype) TYPE_CPLUS_SPECIFIC(thistype)->fn_fieldlists
#define TYPE_FN_FIELDLIST(thistype, n) TYPE_CPLUS_SPECIFIC(thistype)->fn_fieldlists[n]
#define TYPE_FN_FIELDLIST1(thistype, n) TYPE_CPLUS_SPECIFIC(thistype)->fn_fieldlists[n].fn_fields
#define TYPE_FN_FIELDLIST_NAME(thistype, n) TYPE_CPLUS_SPECIFIC(thistype)->fn_fieldlists[n].name
#define TYPE_FN_FIELDLIST_LENGTH(thistype, n) TYPE_CPLUS_SPECIFIC(thistype)->fn_fieldlists[n].length

#define TYPE_N_TEMPLATE_ARGUMENTS(thistype) \
  TYPE_CPLUS_SPECIFIC (thistype)->n_template_arguments
#define TYPE_TEMPLATE_ARGUMENTS(thistype) \
  TYPE_CPLUS_SPECIFIC (thistype)->template_arguments
#define TYPE_TEMPLATE_ARGUMENT(thistype, n) \
  TYPE_CPLUS_SPECIFIC (thistype)->template_arguments[n]

#define TYPE_FN_FIELD(thisfn, n) (thisfn)[n]
#define TYPE_FN_FIELD_PHYSNAME(thisfn, n) (thisfn)[n].physname
#define TYPE_FN_FIELD_TYPE(thisfn, n) (thisfn)[n].type
#define TYPE_FN_FIELD_ARGS(thisfn, n) TYPE_FIELDS ((thisfn)[n].type)
#define TYPE_FN_FIELD_CONST(thisfn, n) ((thisfn)[n].is_const)
#define TYPE_FN_FIELD_VOLATILE(thisfn, n) ((thisfn)[n].is_volatile)
#define TYPE_FN_FIELD_PRIVATE(thisfn, n) ((thisfn)[n].is_private)
#define TYPE_FN_FIELD_PROTECTED(thisfn, n) ((thisfn)[n].is_protected)
#define TYPE_FN_FIELD_PUBLIC(thisfn, n) ((thisfn)[n].is_public)
#define TYPE_FN_FIELD_STATIC(thisfn, n) ((thisfn)[n].is_static)
#define TYPE_FN_FIELD_FINAL(thisfn, n) ((thisfn)[n].is_final)
#define TYPE_FN_FIELD_SYNCHRONIZED(thisfn, n) ((thisfn)[n].is_synchronized)
#define TYPE_FN_FIELD_NATIVE(thisfn, n) ((thisfn)[n].is_native)
#define TYPE_FN_FIELD_ARTIFICIAL(thisfn, n) ((thisfn)[n].is_artificial)
#define TYPE_FN_FIELD_ABSTRACT(thisfn, n) ((thisfn)[n].is_abstract)
#define TYPE_FN_FIELD_STUB(thisfn, n) ((thisfn)[n].is_stub)
#define TYPE_FN_FIELD_CONSTRUCTOR(thisfn, n) ((thisfn)[n].is_constructor)
#define TYPE_FN_FIELD_FCONTEXT(thisfn, n) ((thisfn)[n].fcontext)
#define TYPE_FN_FIELD_VOFFSET(thisfn, n) ((thisfn)[n].voffset-2)
#define TYPE_FN_FIELD_VIRTUAL_P(thisfn, n) ((thisfn)[n].voffset > 1)
#define TYPE_FN_FIELD_STATIC_P(thisfn, n) ((thisfn)[n].voffset == VOFFSET_STATIC)

#define TYPE_TYPEDEF_FIELD_ARRAY(thistype) \
  TYPE_CPLUS_SPECIFIC (thistype)->typedef_field
#define TYPE_TYPEDEF_FIELD(thistype, n) \
  TYPE_CPLUS_SPECIFIC (thistype)->typedef_field[n]
#define TYPE_TYPEDEF_FIELD_NAME(thistype, n) \
  TYPE_TYPEDEF_FIELD (thistype, n).name
#define TYPE_TYPEDEF_FIELD_TYPE(thistype, n) \
  TYPE_TYPEDEF_FIELD (thistype, n).type
#define TYPE_TYPEDEF_FIELD_COUNT(thistype) \
  TYPE_CPLUS_SPECIFIC (thistype)->typedef_field_count

#define TYPE_IS_OPAQUE(thistype) \
  (((TYPE_CODE (thistype) == TYPE_CODE_STRUCT) \
    || (TYPE_CODE (thistype) == TYPE_CODE_UNION)) \
   && (TYPE_NFIELDS (thistype) == 0) \
   && (!HAVE_CPLUS_STRUCT (thistype) \
       || TYPE_NFN_FIELDS (thistype) == 0) \
   && (TYPE_STUB (thistype) || !TYPE_STUB_SUPPORTED (thistype)))

/* A helper macro that returns the name of a type or "unnamed type" if the type
   has no name.  */
#define TYPE_SAFE_NAME(type) \
  (TYPE_NAME (type) ? TYPE_NAME (type) : _("<unnamed type>"))

/* A helper macro that returns the name of an error type.  If the type
   has a name, it is used; otherwise, a default is used.  */
#define TYPE_ERROR_NAME(type) \
  (TYPE_NAME (type) ? TYPE_NAME (type) : _("<error type>"))

struct builtin_type
{
  /* Integral types.  */

  /* Implicit size/sign (based on the architecture's ABI).  */
  struct type *builtin_void;
  struct type *builtin_char;
  struct type *builtin_short;
  struct type *builtin_int;
  struct type *builtin_long;
  struct type *builtin_signed_char;
  struct type *builtin_unsigned_char;
  struct type *builtin_unsigned_short;
  struct type *builtin_unsigned_int;
  struct type *builtin_unsigned_long;
  struct type *builtin_float;
  struct type *builtin_double;
  struct type *builtin_long_double;
  struct type *builtin_complex;
  struct type *builtin_double_complex;
  struct type *builtin_string;
  struct type *builtin_bool;
  struct type *builtin_long_long;
  struct type *builtin_unsigned_long_long;
  struct type *builtin_decfloat;
  struct type *builtin_decdouble;
  struct type *builtin_declong;

  /* "True" character types.
      We use these for the '/c' print format, because c_char is just a
      one-byte integral type, which languages less laid back than C
      will print as ... well, a one-byte integral type.  */
  struct type *builtin_true_char;
  struct type *builtin_true_unsigned_char;

  /* Explicit sizes - see C9X <intypes.h> for naming scheme.  The "int0"
     is for when an architecture needs to describe a register that has
     no size.  */
  struct type *builtin_int0;
  struct type *builtin_int8;
  struct type *builtin_uint8;
  struct type *builtin_int16;
  struct type *builtin_uint16;
  struct type *builtin_int32;
  struct type *builtin_uint32;
  struct type *builtin_int64;
  struct type *builtin_uint64;
  struct type *builtin_int128;
  struct type *builtin_uint128;

  /* Wide character types.  */
  struct type *builtin_char16;
  struct type *builtin_char32;

  /* Pointer types.  */

  /* `pointer to data' type.  Some target platforms use an implicitly
     {sign,zero} -extended 32-bit ABI pointer on a 64-bit ISA.  */
  struct type *builtin_data_ptr;

  /* `pointer to function (returning void)' type.  Harvard
     architectures mean that ABI function and code pointers are not
     interconvertible.  Similarly, since ANSI, C standards have
     explicitly said that pointers to functions and pointers to data
     are not interconvertible --- that is, you can't cast a function
     pointer to void * and back, and expect to get the same value.
     However, all function pointer types are interconvertible, so void
     (*) () can server as a generic function pointer.  */
  struct type *builtin_func_ptr;

  /* `function returning pointer to function (returning void)' type.
     The final void return type is not significant for it.  */
  struct type *builtin_func_func;


  /* Special-purpose types.  */

  /* This type is used to represent a GDB internal function.  */
  struct type *internal_fn;
};

/* Return the type table for the specified architecture.  */
extern const struct builtin_type *builtin_type (struct gdbarch *gdbarch);


/* Per-objfile types used by symbol readers.  */

struct objfile_type
{
  /* Basic types based on the objfile architecture.  */
  struct type *builtin_void;
  struct type *builtin_char;
  struct type *builtin_short;
  struct type *builtin_int;
  struct type *builtin_long;
  struct type *builtin_long_long;
  struct type *builtin_signed_char;
  struct type *builtin_unsigned_char;
  struct type *builtin_unsigned_short;
  struct type *builtin_unsigned_int;
  struct type *builtin_unsigned_long;
  struct type *builtin_unsigned_long_long;
  struct type *builtin_float;
  struct type *builtin_double;
  struct type *builtin_long_double;

  /* This type is used to represent symbol addresses.  */
  struct type *builtin_core_addr;

  /* This type represents a type that was unrecognized in symbol read-in.  */
  struct type *builtin_error;

  /* Types used for symbols with no debug information.  */
  struct type *nodebug_text_symbol;
  struct type *nodebug_text_gnu_ifunc_symbol;
  struct type *nodebug_got_plt_symbol;
  struct type *nodebug_data_symbol;
  struct type *nodebug_unknown_symbol;
  struct type *nodebug_tls_symbol;
};

/* Return the type table for the specified objfile.  */
extern const struct objfile_type *objfile_type (struct objfile *objfile);

 
/* Explicit floating-point formats.  See "floatformat.h".  */
extern const struct floatformat *floatformats_ieee_half[BFD_ENDIAN_UNKNOWN];
extern const struct floatformat *floatformats_ieee_single[BFD_ENDIAN_UNKNOWN];
extern const struct floatformat *floatformats_ieee_double[BFD_ENDIAN_UNKNOWN];
extern const struct floatformat *floatformats_ieee_double_littlebyte_bigword[BFD_ENDIAN_UNKNOWN];
extern const struct floatformat *floatformats_i387_ext[BFD_ENDIAN_UNKNOWN];
extern const struct floatformat *floatformats_m68881_ext[BFD_ENDIAN_UNKNOWN];
extern const struct floatformat *floatformats_arm_ext[BFD_ENDIAN_UNKNOWN];
extern const struct floatformat *floatformats_ia64_spill[BFD_ENDIAN_UNKNOWN];
extern const struct floatformat *floatformats_ia64_quad[BFD_ENDIAN_UNKNOWN];
extern const struct floatformat *floatformats_vax_f[BFD_ENDIAN_UNKNOWN];
extern const struct floatformat *floatformats_vax_d[BFD_ENDIAN_UNKNOWN];
extern const struct floatformat *floatformats_ibm_long_double[BFD_ENDIAN_UNKNOWN];


/* Allocate space for storing data associated with a particular type.
   We ensure that the space is allocated using the same mechanism that
   was used to allocate the space for the type structure itself.  I.e.
   if the type is on an objfile's objfile_obstack, then the space for data
   associated with that type will also be allocated on the objfile_obstack.
   If the type is not associated with any particular objfile (such as
   builtin types), then the data space will be allocated with xmalloc,
   the same as for the type structure.  */

#define TYPE_ALLOC(t,size)  \
   (TYPE_OBJFILE_OWNED (t) \
    ? obstack_alloc (&TYPE_OBJFILE (t) -> objfile_obstack, size) \
    : xmalloc (size))

#define TYPE_ZALLOC(t,size)  \
   (TYPE_OBJFILE_OWNED (t) \
    ? memset (obstack_alloc (&TYPE_OBJFILE (t)->objfile_obstack, size),  \
	      0, size)  \
    : xzalloc (size))

/* Use alloc_type to allocate a type owned by an objfile.
   Use alloc_type_arch to allocate a type owned by an architecture.
   Use alloc_type_copy to allocate a type with the same owner as a
   pre-existing template type, no matter whether objfile or gdbarch.  */
extern struct type *alloc_type (struct objfile *);
extern struct type *alloc_type_arch (struct gdbarch *);
extern struct type *alloc_type_copy (const struct type *);

/* Return the type's architecture.  For types owned by an architecture,
   that architecture is returned.  For types owned by an objfile, that
   objfile's architecture is returned.  */
extern struct gdbarch *get_type_arch (const struct type *);

/* Helper function to construct objfile-owned types.  */
extern struct type *init_type (enum type_code, int, int, const char *,
			       struct objfile *);

/* Helper functions to construct architecture-owned types.  */
extern struct type *arch_type (struct gdbarch *, enum type_code, int, char *);
extern struct type *arch_integer_type (struct gdbarch *, int, int, char *);
extern struct type *arch_character_type (struct gdbarch *, int, int, char *);
extern struct type *arch_boolean_type (struct gdbarch *, int, int, char *);
extern struct type *arch_float_type (struct gdbarch *, int, char *,
				     const struct floatformat **);
extern struct type *arch_complex_type (struct gdbarch *, char *,
				       struct type *);

/* Helper functions to construct a struct or record type.  An
   initially empty type is created using arch_composite_type().
   Fields are then added using append_composite_type_field*().  A union
   type has its size set to the largest field.  A struct type has each
   field packed against the previous.  */

extern struct type *arch_composite_type (struct gdbarch *gdbarch,
					 char *name, enum type_code code);
extern void append_composite_type_field (struct type *t, char *name,
					 struct type *field);
extern void append_composite_type_field_aligned (struct type *t,
						 char *name,
						 struct type *field,
						 int alignment);
struct field *append_composite_type_field_raw (struct type *t, char *name,
					       struct type *field);

/* Helper functions to construct a bit flags type.  An initially empty
   type is created using arch_flag_type().  Flags are then added using
   append_flag_type_flag().  */
extern struct type *arch_flags_type (struct gdbarch *gdbarch,
				     char *name, int length);
extern void append_flags_type_flag (struct type *type, int bitpos, char *name);

extern void make_vector_type (struct type *array_type);
extern struct type *init_vector_type (struct type *elt_type, int n);

extern struct type *lookup_reference_type (struct type *);

extern struct type *make_reference_type (struct type *, struct type **);

extern struct type *make_cv_type (int, int, struct type *, struct type **);

extern struct type *make_restrict_type (struct type *);

extern void replace_type (struct type *, struct type *);

extern int address_space_name_to_int (struct gdbarch *, char *);

extern const char *address_space_int_to_name (struct gdbarch *, int);

extern struct type *make_type_with_address_space (struct type *type, 
						  int space_identifier);

extern struct type *lookup_memberptr_type (struct type *, struct type *);

extern struct type *lookup_methodptr_type (struct type *);

extern void smash_to_method_type (struct type *type, struct type *domain,
				  struct type *to_type, struct field *args,
				  int nargs, int varargs);

extern void smash_to_memberptr_type (struct type *, struct type *,
				     struct type *);

extern void smash_to_methodptr_type (struct type *, struct type *);

extern struct type *allocate_stub_method (struct type *);

extern const char *type_name_no_tag (const struct type *);

extern const char *type_name_no_tag_or_error (struct type *type);

extern struct type *lookup_struct_elt_type (struct type *, char *, int);

extern struct type *make_pointer_type (struct type *, struct type **);

extern struct type *lookup_pointer_type (struct type *);

extern struct type *make_function_type (struct type *, struct type **);

extern struct type *lookup_function_type (struct type *);

extern struct type *lookup_function_type_with_arguments (struct type *,
							 int,
							 struct type **);

extern struct type *create_range_type (struct type *, struct type *, LONGEST,
				       LONGEST);
 
extern struct type *create_range_type_d (struct type *, struct type *,
				        int, int, void *, void *,  void*,
					LONGEST (*)(void*, CORE_ADDR, void*));

extern struct type * create_range_type_d_pgi (struct type *result_type, struct type *index_type,
					    int low_bound, int high_bound, int stride, int soffset, int lstride,
					    void *dwarf_low, void *dwarf_high, void *dwarf_count,
					    void *dwarf_stride, void *dwarf_soffset, void *dwarf_lstride,
					    LONGEST (*expr_evaluate)(void*, CORE_ADDR, void*));

extern struct type *create_range_type_d (struct type *, struct type *,
					 int, int, void *, void *,  void*,
					 LONGEST (*)(void*, CORE_ADDR, void*));

extern struct type * create_range_type_d_pgi (struct type *result_type, struct type *index_type,
		         int low_bound, int high_bound, int stride, int soffset, int lstride,
                         void *dwarf_low, void *dwarf_high, void *dwarf_count,
                         void *dwarf_stride, void *dwarf_soffset, void *dwarf_lstride,
		         LONGEST (*expr_evaluate)(void*, CORE_ADDR, void*));

extern struct type *create_array_type (struct type *, struct type *,
				       struct type *);
extern struct type *lookup_array_range_type (struct type *, LONGEST, LONGEST);

extern struct type *create_string_type (struct type *, struct type *,
					struct type *);
extern struct type *lookup_string_range_type (struct type *, LONGEST, LONGEST);

extern struct type *create_set_type (struct type *, struct type *);

extern struct type *lookup_unsigned_typename (const struct language_defn *,
					      struct gdbarch *, const char *);

extern struct type *lookup_signed_typename (const struct language_defn *,
					    struct gdbarch *, const char *);

extern struct type *check_typedef (struct type *);

#define CHECK_TYPEDEF(TYPE)			\
  do {						\
    (TYPE) = check_typedef (TYPE);		\
  } while (0)

extern void check_stub_method_group (struct type *, int);

extern char *gdb_mangle_name (struct type *, int, int);

extern struct type *lookup_typename (const struct language_defn *,
				     struct gdbarch *, const char *,
				     const struct block *, int);

extern struct type *lookup_template_type (char *, struct type *,
					  const struct block *);

extern int get_vptr_fieldno (struct type *, struct type **);

extern int get_discrete_bounds (struct type *, LONGEST *, LONGEST *);

extern int get_array_bounds (struct type *type, LONGEST *low_bound,
			     LONGEST *high_bound);

extern int class_types_same_p (const struct type *, const struct type *);

extern int is_ancestor (struct type *, struct type *);

extern int is_public_ancestor (struct type *, struct type *);

extern int is_unique_ancestor (struct type *, struct value *);

extern int range_is_co_shape_p (struct type *);

/* Overload resolution */

#define LENGTH_MATCH(bv) ((bv)->rank[0])

/* Badness if parameter list length doesn't match arg list length.  */
extern const struct rank LENGTH_MISMATCH_BADNESS;

/* Dummy badness value for nonexistent parameter positions.  */
extern const struct rank TOO_FEW_PARAMS_BADNESS;
/* Badness if no conversion among types.  */
extern const struct rank INCOMPATIBLE_TYPE_BADNESS;

/* Badness of an exact match.  */
extern const struct rank EXACT_MATCH_BADNESS;

/* Badness of integral promotion.  */
extern const struct rank INTEGER_PROMOTION_BADNESS;
/* Badness of floating promotion.  */
extern const struct rank FLOAT_PROMOTION_BADNESS;
/* Badness of converting a derived class pointer
   to a base class pointer.  */
extern const struct rank BASE_PTR_CONVERSION_BADNESS;
/* Badness of integral conversion.  */
extern const struct rank INTEGER_CONVERSION_BADNESS;
/* Badness of floating conversion.  */
extern const struct rank FLOAT_CONVERSION_BADNESS;
/* Badness of integer<->floating conversions.  */
extern const struct rank INT_FLOAT_CONVERSION_BADNESS;
/* Badness of conversion of pointer to void pointer.  */
extern const struct rank VOID_PTR_CONVERSION_BADNESS;
/* Badness of conversion to boolean.  */
extern const struct rank BOOL_CONVERSION_BADNESS;
/* Badness of converting derived to base class.  */
extern const struct rank BASE_CONVERSION_BADNESS;
/* Badness of converting from non-reference to reference.  */
extern const struct rank REFERENCE_CONVERSION_BADNESS;
/* Badness of converting integer 0 to NULL pointer.  */
extern const struct rank NULL_POINTER_CONVERSION;

/* Non-standard conversions allowed by the debugger */
/* Converting a pointer to an int is usually OK.  */
extern const struct rank NS_POINTER_CONVERSION_BADNESS;

/* Badness of converting a (non-zero) integer constant
   to a pointer.  */
extern const struct rank NS_INTEGER_POINTER_CONVERSION_BADNESS;

extern struct rank sum_ranks (struct rank a, struct rank b);
extern int compare_ranks (struct rank a, struct rank b);

extern int compare_badness (struct badness_vector *, struct badness_vector *);

extern struct badness_vector *rank_function (struct type **, int,
					     struct value **, int);

extern struct rank rank_one_type (struct type *, struct type *,
				  struct value *);

extern void recursive_dump_type (struct type *, int);

extern int field_is_static (struct field *);

/* printcmd.c */

extern void print_scalar_formatted (const void *, struct type *,
				    const struct value_print_options *,
				    int, struct ui_file *);

extern int can_dereference (struct type *);

extern int is_integral_type (struct type *);

extern int is_scalar_type_recursive (struct type *);

extern void maintenance_print_type (char *, int);

extern htab_t create_copied_types_hash (struct objfile *objfile);

extern struct type *copy_type_recursive (struct objfile *objfile,
					 struct type *type,
					 htab_t copied_types);

extern struct type *copy_type (const struct type *type);

extern int types_equal (struct type *, struct type *);

/* CUDA - find the target type with the same instance flags */
struct type *find_target_type_with_instance_flags (struct type *type);

#endif /* GDBTYPES_H */
