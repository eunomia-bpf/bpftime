/* GDB-friendly replacement for <assert.h>.
   Copyright (C) 2000-2013 Free Software Foundation, Inc.

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

#ifndef GDB_ASSERT_H
#define GDB_ASSERT_H

/* A static assertion.  This will cause a compile-time error if EXPR,
   which must be a compile-time constant, is false.  */

#define gdb_static_assert(expr) \
  extern int never_defined_just_used_for_checking[(expr) ? 1 : -1]

/* PRAGMATICS: "gdb_assert.h":gdb_assert() is a lower case (rather
   than upper case) macro since that provides the closest fit to the
   existing lower case macro <assert.h>:assert() that it is
   replacing.  */

#define gdb_assert(expr)                                                      \
  ((void) ((expr) ? 0 :                                                       \
	   (gdb_assert_fail (#expr, __FILE__, __LINE__, ASSERT_FUNCTION), 0)))

/* Version 2.4 and later of GCC define a magical variable `__PRETTY_FUNCTION__'
   which contains the name of the function currently being defined.
   This is broken in G++ before version 2.6.
   C9x has a similar variable called __func__, but prefer the GCC one since
   it demangles C++ function names.  */
#if (GCC_VERSION >= 2004)
#define ASSERT_FUNCTION		__PRETTY_FUNCTION__
#else
#if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#define ASSERT_FUNCTION		__func__
#endif
#endif

/* This prints an "Assertion failed" message, asking the user if they
   want to continue, dump core, or just exit.  */
#if defined (ASSERT_FUNCTION)
#define gdb_assert_fail(assertion, file, line, function)                      \
  internal_error (file, line, _("%s: Assertion `%s' failed."),                \
		  function, assertion)
#else
#define gdb_assert_fail(assertion, file, line, function)                      \
  internal_error (file, line, _("Assertion `%s' failed."),                    \
		  assertion)
#endif

/* The canonical form of gdb_assert (0).
   MESSAGE is a string to include in the error message.  */

#if defined (ASSERT_FUNCTION)
#define gdb_assert_not_reached(message) \
  internal_error (__FILE__, __LINE__, "%s: %s", ASSERT_FUNCTION, _(message))
#else
#define gdb_assert_not_reached(message) \
  internal_error (__FILE__, __LINE__, _(message))
#endif

#endif /* gdb_assert.h */
