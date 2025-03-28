/* Manages interpreters for GDB, the GNU debugger.

   Copyright (C) 2000-2013 Free Software Foundation, Inc.

   Written by Jim Ingham <jingham@apple.com> of Apple Computer, Inc.

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

#ifndef INTERPS_H
#define INTERPS_H

#include "exceptions.h"

struct ui_out;
struct interp;

extern int interp_resume (struct interp *interp);
extern int interp_suspend (struct interp *interp);
extern int interp_prompt_p (struct interp *interp);
extern int interp_exec_p (struct interp *interp);
extern struct gdb_exception interp_exec (struct interp *interp,
					 const char *command);
extern int interp_quiet_p (struct interp *interp);

typedef void *(interp_init_ftype) (struct interp *self, int top_level);
typedef int (interp_resume_ftype) (void *data);
typedef int (interp_suspend_ftype) (void *data);
typedef int (interp_prompt_p_ftype) (void *data);
typedef struct gdb_exception (interp_exec_ftype) (void *data,
						  const char *command);
typedef void (interp_command_loop_ftype) (void *data);
typedef struct ui_out *(interp_ui_out_ftype) (struct interp *self);

typedef int (interp_set_logging_ftype) (struct interp *self, int start_log,
					struct ui_file *out,
					struct ui_file *logfile);

struct interp_procs
{
  interp_init_ftype *init_proc;
  interp_resume_ftype *resume_proc;
  interp_suspend_ftype *suspend_proc;
  interp_exec_ftype *exec_proc;
  interp_prompt_p_ftype *prompt_proc_p;

  /* Returns the ui_out currently used to collect results for this
     interpreter.  It can be a formatter for stdout, as is the case
     for the console & mi outputs, or it might be a result
     formatter.  */
  interp_ui_out_ftype *ui_out_proc;

  /* Provides a hook for interpreters to do any additional
     setup/cleanup that they might need when logging is enabled or
     disabled.  */
  interp_set_logging_ftype *set_logging_proc;

  interp_command_loop_ftype *command_loop_proc;
};

extern struct interp *interp_new (const char *name, const struct interp_procs *procs);
extern void interp_add (struct interp *interp);
extern int interp_set (struct interp *interp, int top_level);
extern struct interp *interp_lookup (const char *name);
extern struct ui_out *interp_ui_out (struct interp *interp);
extern void *interp_data (struct interp *interp);
extern const char *interp_name (struct interp *interp);
extern struct interp *interp_set_temp (const char *name);

extern int current_interp_named_p (const char *name);
extern int current_interp_display_prompt_p (void);
extern void current_interp_command_loop (void);

/* Call this function to give the current interpreter an opportunity
   to do any special handling of streams when logging is enabled or
   disabled.  START_LOG is 1 when logging is starting, 0 when it ends,
   and OUT is the stream for the log file; it will be NULL when
   logging is ending.  LOGFILE is non-NULL if the output streams
   are to be tees, with the log file as one of the outputs.  */

extern int current_interp_set_logging (int start_log, struct ui_file *out,
				       struct ui_file *logfile);

/* Returns opaque data associated with the top-level interpreter.  */
extern void *top_level_interpreter_data (void);
extern struct interp *top_level_interpreter (void);

/* True if the current interpreter is in async mode, false if in sync
   mode.  If in sync mode, running a synchronous execution command
   (with execute_command, e.g, "next") will not return until the
   command is finished.  If in async mode, then running a synchronous
   command returns right after resuming the target.  Waiting for the
   command's completion is later done on the top event loop (using
   continuations).  */
extern int interpreter_async;

extern void clear_interpreter_hooks (void);

/* well-known interpreters */
#define INTERP_CONSOLE		"console"
#define INTERP_MI1             "mi1"
#define INTERP_MI2             "mi2"
#define INTERP_MI3             "mi3"
#define INTERP_MI		"mi"
#define INTERP_TUI		"tui"
#define INTERP_INSIGHT		"insight"

#endif
