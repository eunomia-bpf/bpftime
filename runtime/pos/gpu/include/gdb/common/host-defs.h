/* Basic host-specific definitions for GDB.
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

#ifndef HOST_DEFS_H
#define HOST_DEFS_H

#ifdef __MSDOS__
# define CANT_FORK
# define GLOBAL_CURDIR
# define DIRNAME_SEPARATOR ';'
#endif

#if !defined (__CYGWIN__) && defined (_WIN32)
# define DIRNAME_SEPARATOR ';'
#endif

#ifndef DIRNAME_SEPARATOR
#define DIRNAME_SEPARATOR ':'
#endif

#ifndef SLASH_STRING
#define SLASH_STRING "/"
#endif

#endif /* HOST_DEFS_H */
