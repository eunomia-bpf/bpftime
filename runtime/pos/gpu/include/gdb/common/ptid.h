/* The ptid_t type and common functions operating on it.

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

#ifndef PTID_H
#define PTID_H

/* The ptid struct is a collection of the various "ids" necessary
   for identifying the inferior.  This consists of the process id
   (pid), thread id (tid), and other fields necessary for uniquely
   identifying the inferior process/thread being debugged.  When
   manipulating ptids, the constructors, accessors, and predicate
   declared in server.h should be used.  These are as follows:

      ptid_build	- Make a new ptid from a pid, lwp, and tid.
      pid_to_ptid	- Make a new ptid from just a pid.
      ptid_get_pid	- Fetch the pid component of a ptid.
      ptid_get_lwp	- Fetch the lwp component of a ptid.
      ptid_get_tid	- Fetch the tid component of a ptid.
      ptid_equal	- Test to see if two ptids are equal.

   Please do NOT access the struct ptid members directly (except, of
   course, in the implementation of the above ptid manipulation
   functions).  */

struct ptid
  {
    /* Process id */
    int pid;

    /* Lightweight process id */
    long lwp;

    /* Thread id */
    long tid;
  };

typedef struct ptid ptid_t;

/* The null or zero ptid, often used to indicate no process. */
extern ptid_t null_ptid;

/* The -1 ptid, often used to indicate either an error condition
   or a "don't care" condition, i.e, "run all threads."  */
extern ptid_t minus_one_ptid;

/* Attempt to find and return an existing ptid with the given PID, LWP,
   and TID components.  If none exists, create a new one and return
   that.  */
ptid_t ptid_build (int pid, long lwp, long tid);

/* Find/Create a ptid from just a pid. */
ptid_t pid_to_ptid (int pid);

/* Fetch the pid (process id) component from a ptid. */
int ptid_get_pid (ptid_t ptid);

/* Fetch the lwp (lightweight process) component from a ptid. */
long ptid_get_lwp (ptid_t ptid);

/* Fetch the tid (thread id) component from a ptid. */
long ptid_get_tid (ptid_t ptid);

/* Compare two ptids to see if they are equal */
int ptid_equal (ptid_t p1, ptid_t p2);

/* Return true if PTID represents a process id.  */
int ptid_is_pid (ptid_t ptid);

#endif
