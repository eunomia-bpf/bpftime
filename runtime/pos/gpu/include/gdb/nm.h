/* Native support for GNU/Linux.

   Copyright (C) 1999-2013 Free Software Foundation, Inc.

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

/* Use elf_gregset_t and elf_fpregset_t, rather than
   gregset_t and fpregset_t.  */

#define GDB_GREGSET_T  elf_gregset_t
#define GDB_FPREGSET_T elf_fpregset_t
