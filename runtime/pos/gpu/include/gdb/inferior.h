/* Variables that describe the inferior process running under GDB:
   Where it is, why it stopped, and how to step it.

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

#if !defined (INFERIOR_H)
#define INFERIOR_H 1

struct target_waitstatus;
struct frame_info;
struct ui_file;
struct type;
struct gdbarch;
struct regcache;
struct ui_out;
struct terminal_info;
struct target_desc_info;

#include "ptid.h"

/* For bpstat.  */
#include "breakpoint.h"

/* For enum gdb_signal.  */
#include "target.h"

/* For struct frame_id.  */
#include "frame.h"

#include "progspace.h"
#include "registry.h"

struct infcall_suspend_state;
struct infcall_control_state;

extern struct infcall_suspend_state *save_infcall_suspend_state (void);
extern struct infcall_control_state *save_infcall_control_state (void);

extern void restore_infcall_suspend_state (struct infcall_suspend_state *);
extern void restore_infcall_control_state (struct infcall_control_state *);

extern struct cleanup *make_cleanup_restore_infcall_suspend_state
					    (struct infcall_suspend_state *);
extern struct cleanup *make_cleanup_restore_infcall_control_state
					    (struct infcall_control_state *);

extern void discard_infcall_suspend_state (struct infcall_suspend_state *);
extern void discard_infcall_control_state (struct infcall_control_state *);

extern struct regcache *
  get_infcall_suspend_state_regcache (struct infcall_suspend_state *);

/* Returns true if PTID matches filter FILTER.  FILTER can be the wild
   card MINUS_ONE_PTID (all ptid match it); can be a ptid representing
   a process (ptid_is_pid returns true), in which case, all lwps and
   threads of that given process match, lwps and threads of other
   processes do not; or, it can represent a specific thread, in which
   case, only that thread will match true.  PTID must represent a
   specific LWP or THREAD, it can never be a wild card.  */

extern int ptid_match (ptid_t ptid, ptid_t filter);

/* Save value of inferior_ptid so that it may be restored by
   a later call to do_cleanups().  Returns the struct cleanup
   pointer needed for later doing the cleanup.  */
extern struct cleanup * save_inferior_ptid (void);

extern void set_sigint_trap (void);

extern void clear_sigint_trap (void);

/* Set/get file name for default use for standard in/out in the inferior.  */

extern void set_inferior_io_terminal (const char *terminal_name);
extern const char *get_inferior_io_terminal (void);

/* Collected pid, tid, etc. of the debugged inferior.  When there's
   no inferior, PIDGET (inferior_ptid) will be 0.  */

extern ptid_t inferior_ptid;

/* Are we simulating synchronous execution? This is used in async gdb
   to implement the 'run', 'continue' etc commands, which will not
   redisplay the prompt until the execution is actually over.  */
extern int sync_execution;

/* Inferior environment.  */

extern void clear_proceed_status (void);

extern void proceed (CORE_ADDR, enum gdb_signal, int);

extern int sched_multi;

/* When set, stop the 'step' command if we enter a function which has
   no line number information.  The normal behavior is that we step
   over such function.  */
extern int step_stop_if_no_debug;

/* If set, the inferior should be controlled in non-stop mode.  In
   this mode, each thread is controlled independently.  Execution
   commands apply only to the selected thread by default, and stop
   events stop only the thread that had the event -- the other threads
   are kept running freely.  */
extern int non_stop;

/* If set (default), when following a fork, GDB will detach from one
   the fork branches, child or parent.  Exactly which branch is
   detached depends on 'set follow-fork-mode' setting.  */
extern int detach_fork;

/* When set (default), the target should attempt to disable the operating
   system's address space randomization feature when starting an inferior.  */
extern int disable_randomization;

extern void generic_mourn_inferior (void);

extern void terminal_save_ours (void);

extern void terminal_ours (void);

extern CORE_ADDR unsigned_pointer_to_address (struct gdbarch *gdbarch,
					      struct type *type,
					      const gdb_byte *buf);
extern void unsigned_address_to_pointer (struct gdbarch *gdbarch,
					 struct type *type, gdb_byte *buf,
					 CORE_ADDR addr);
extern CORE_ADDR signed_pointer_to_address (struct gdbarch *gdbarch,
					    struct type *type,
					    const gdb_byte *buf);
extern void address_to_signed_pointer (struct gdbarch *gdbarch,
				       struct type *type, gdb_byte *buf,
				       CORE_ADDR addr);

extern void wait_for_inferior (void);

extern void prepare_for_detach (void);

extern void fetch_inferior_event (void *);

extern void init_wait_for_inferior (void);

extern void reopen_exec_file (void);

/* The `resume' routine should only be called in special circumstances.
   Normally, use `proceed', which handles a lot of bookkeeping.  */

extern void resume (int, enum gdb_signal);

extern ptid_t user_visible_resume_ptid (int step);

extern void insert_step_resume_breakpoint_at_sal (struct gdbarch *,
						  struct symtab_and_line ,
						  struct frame_id);

/* From misc files */

extern void default_print_registers_info (struct gdbarch *gdbarch,
					  struct ui_file *file,
					  struct frame_info *frame,
					  int regnum, int all);

extern void child_terminal_info (char *, int);

extern void term_info (char *, int);

extern void terminal_ours_for_output (void);

extern void terminal_inferior (void);

extern void terminal_init_inferior (void);

extern void terminal_init_inferior_with_pgrp (int pgrp);

/* From fork-child.c */

extern int fork_inferior (char *, char *, char **,
			  void (*)(void),
			  void (*)(int), void (*)(void), char *,
                          void (*)(const char *,
                                   char * const *, char * const *));


extern void startup_inferior (int);

extern char *construct_inferior_arguments (int, char **);

/* From infrun.c */

extern unsigned int debug_infrun;

extern int stop_on_solib_events;

extern void start_remote (int from_tty);

extern void normal_stop (void);

extern int signal_stop_state (int);

extern int signal_print_state (int);

extern int signal_pass_state (int);

extern int signal_stop_update (int, int);

extern int signal_print_update (int, int);

extern int signal_pass_update (int, int);

extern void get_last_target_status(ptid_t *ptid,
                                   struct target_waitstatus *status);

extern void follow_inferior_reset_breakpoints (void);

void set_step_info (struct frame_info *frame, struct symtab_and_line sal);

extern void insert_step_resume_breakpoint_at_caller (struct frame_info *);

/* From infcmd.c */

extern void post_create_inferior (struct target_ops *, int);

extern void attach_command (char *, int);

extern char *get_inferior_args (void);

extern void set_inferior_args (char *);

extern void set_inferior_args_vector (int, char **);

extern void registers_info (char *, int);

extern void continue_1 (int all_threads);

extern void interrupt_target_1 (int all_threads);

extern void delete_longjmp_breakpoint_cleanup (void *arg);

extern void detach_command (char *, int);

extern void notice_new_inferior (ptid_t, int, int);

extern struct value *get_return_value (struct value *function,
                                       struct type *value_type);

/* Address at which inferior stopped.  */

extern CORE_ADDR stop_pc;

/* Nonzero if stopped due to completion of a stack dummy routine.  */

extern enum stop_stack_kind stop_stack_dummy;

/* Nonzero if program stopped due to a random (unexpected) signal in
   inferior process.  */

extern int stopped_by_random_signal;

/* STEP_OVER_ALL means step over all subroutine calls.
   STEP_OVER_UNDEBUGGABLE means step over calls to undebuggable functions.
   STEP_OVER_NONE means don't step over any subroutine calls.  */

enum step_over_calls_kind
  {
    STEP_OVER_NONE,
    STEP_OVER_ALL,
    STEP_OVER_UNDEBUGGABLE
  };

/* Anything but NO_STOP_QUIETLY means we expect a trap and the caller
   will handle it themselves.  STOP_QUIETLY is used when running in
   the shell before the child program has been exec'd and when running
   through shared library loading.  STOP_QUIETLY_REMOTE is used when
   setting up a remote connection; it is like STOP_QUIETLY_NO_SIGSTOP
   except that there is no need to hide a signal.  */

/* It is also used after attach, due to attaching to a process.  This
   is a bit trickier.  When doing an attach, the kernel stops the
   debuggee with a SIGSTOP.  On newer GNU/Linux kernels (>= 2.5.61)
   the handling of SIGSTOP for a ptraced process has changed.  Earlier
   versions of the kernel would ignore these SIGSTOPs, while now
   SIGSTOP is treated like any other signal, i.e. it is not muffled.
   
   If the gdb user does a 'continue' after the 'attach', gdb passes
   the global variable stop_signal (which stores the signal from the
   attach, SIGSTOP) to the ptrace(PTRACE_CONT,...)  call.  This is
   problematic, because the kernel doesn't ignore such SIGSTOP
   now.  I.e. it is reported back to gdb, which in turn presents it
   back to the user.
 
   To avoid the problem, we use STOP_QUIETLY_NO_SIGSTOP, which allows
   gdb to clear the value of stop_signal after the attach, so that it
   is not passed back down to the kernel.  */

enum stop_kind
  {
    NO_STOP_QUIETLY = 0,
    STOP_QUIETLY,
    STOP_QUIETLY_REMOTE,
    STOP_QUIETLY_NO_SIGSTOP
  };

/* Reverse execution.  */
enum exec_direction_kind
  {
    EXEC_FORWARD,
    EXEC_REVERSE
  };

/* The current execution direction.  This should only be set to enum
   exec_direction_kind values.  It is only an int to make it
   compatible with make_cleanup_restore_integer.  */
extern int execution_direction;

/* Save register contents here when executing a "finish" command or are
   about to pop a stack dummy frame, if-and-only-if proceed_to_finish is set.
   Thus this contains the return value from the called function (assuming
   values are returned in a register).  */

extern struct regcache *stop_registers;

/* True if we are debugging displaced stepping.  */
extern int debug_displaced;

/* Dump LEN bytes at BUF in hex to FILE, followed by a newline.  */
void displaced_step_dump_bytes (struct ui_file *file,
                                const gdb_byte *buf, size_t len);

struct displaced_step_closure *get_displaced_step_closure_by_addr (CORE_ADDR addr);

/* Possible values for gdbarch_call_dummy_location.  */
#define ON_STACK 1
#define AT_ENTRY_POINT 4

/* If STARTUP_WITH_SHELL is set, GDB's "run"
   will attempts to start up the debugee under a shell.
   This is in order for argument-expansion to occur.  E.g.,
   (gdb) run *
   The "*" gets expanded by the shell into a list of files.
   While this is a nice feature, it turns out to interact badly
   with some of the catch-fork/catch-exec features we have added.
   In particular, if the shell does any fork/exec's before
   the exec of the target program, that can confuse GDB.
   To disable this feature, set STARTUP_WITH_SHELL to 0.
   To enable this feature, set STARTUP_WITH_SHELL to 1.
   The catch-exec traps expected during start-up will
   be 1 if target is not started up with a shell, 2 if it is.
   - RT
   If you disable this, you need to decrement
   START_INFERIOR_TRAPS_EXPECTED in tm.h.  */
#define STARTUP_WITH_SHELL 1
#if !defined(START_INFERIOR_TRAPS_EXPECTED)
#define START_INFERIOR_TRAPS_EXPECTED	2
#endif

struct private_inferior;

/* Inferior process specific part of `struct infcall_control_state'.

   Inferior thread counterpart is `struct thread_control_state'.  */

struct inferior_control_state
{
  /* See the definition of stop_kind above.  */
  enum stop_kind stop_soon;
};

/* Inferior process specific part of `struct infcall_suspend_state'.

   Inferior thread counterpart is `struct thread_suspend_state'.  */

#if 0 /* Currently unused and empty structures are not valid C.  */
struct inferior_suspend_state
{
};
#endif

/* GDB represents the state of each program execution with an object
   called an inferior.  An inferior typically corresponds to a process
   but is more general and applies also to targets that do not have a
   notion of processes.  Each run of an executable creates a new
   inferior, as does each attachment to an existing process.
   Inferiors have unique internal identifiers that are different from
   target process ids.  Each inferior may in turn have multiple
   threads running in it.  */

struct inferior
{
  /* Pointer to next inferior in singly-linked list of inferiors.  */
  struct inferior *next;

  /* Convenient handle (GDB inferior id).  Unique across all
     inferiors.  */
  int num;

  /* Actual target inferior id, usually, a process id.  This matches
     the ptid_t.pid member of threads of this inferior.  */
  int pid;
  /* True if the PID was actually faked by GDB.  */
  int fake_pid_p;

  /* State of GDB control of inferior process execution.
     See `struct inferior_control_state'.  */
  struct inferior_control_state control;

  /* State of inferior process to restore after GDB is done with an inferior
     call.  See `struct inferior_suspend_state'.  */
#if 0 /* Currently unused and empty structures are not valid C.  */
  struct inferior_suspend_state suspend;
#endif

  /* True if this was an auto-created inferior, e.g. created from
     following a fork; false, if this inferior was manually added by
     the user, and we should not attempt to prune it
     automatically.  */
  int removable;

  /* The address space bound to this inferior.  */
  struct address_space *aspace;

  /* The program space bound to this inferior.  */
  struct program_space *pspace;

  /* The arguments string to use when running.  */
  char *args;

  /* The size of elements in argv.  */
  int argc;

  /* The vector version of arguments.  If ARGC is nonzero,
     then we must compute ARGS from this (via the target).
     This is always coming from main's argv and therefore
     should never be freed.  */
  char **argv;

  /* The name of terminal device to use for I/O.  */
  char *terminal;

  /* Environment to use for running inferior,
     in format described in environ.h.  */
  struct gdb_environ *environment;

  /* Nonzero if this child process was attached rather than
     forked.  */
  int attach_flag;

  /* If this inferior is a vfork child, then this is the pointer to
     its vfork parent, if GDB is still attached to it.  */
  struct inferior *vfork_parent;

  /* If this process is a vfork parent, this is the pointer to the
     child.  Since a vfork parent is left frozen by the kernel until
     the child execs or exits, a process can only have one vfork child
     at a given time.  */
  struct inferior *vfork_child;

  /* True if this inferior should be detached when it's vfork sibling
     exits or execs.  */
  int pending_detach;

  /* True if this inferior is a vfork parent waiting for a vfork child
     not under our control to be done with the shared memory region,
     either by exiting or execing.  */
  int waiting_for_vfork_done;

  /* True if we're in the process of detaching from this inferior.  */
  int detaching;

  /* What is left to do for an execution command after any thread of
     this inferior stops.  For continuations associated with a
     specific thread, see `struct thread_info'.  */
  struct continuation *continuations;

  /* Private data used by the target vector implementation.  */
  struct private_inferior *private;

  /* HAS_EXIT_CODE is true if the inferior exited with an exit code.
     In this case, the EXIT_CODE field is also valid.  */
  int has_exit_code;
  LONGEST exit_code;

  /* Default flags to pass to the symbol reading functions.  These are
     used whenever a new objfile is created.  The valid values come
     from enum symfile_add_flags.  */
  int symfile_flags;

  /* Info about an inferior's target description (if it's fetched; the
     user supplied description's filename, if any; etc.).  */
  struct target_desc_info *tdesc_info;

  /* The architecture associated with the inferior through the
     connection to the target.

     The architecture vector provides some information that is really
     a property of the inferior, accessed through a particular target:
     ptrace operations; the layout of certain RSP packets; the
     solib_ops vector; etc.  To differentiate architecture accesses to
     per-inferior/target properties from
     per-thread/per-frame/per-objfile properties, accesses to
     per-inferior/target properties should be made through
     this gdbarch.  */
  struct gdbarch *gdbarch;

  /* Per inferior data-pointers required by other GDB modules.  */
  REGISTRY_FIELDS;
};

/* Keep a registry of per-inferior data-pointers required by other GDB
   modules.  */

DECLARE_REGISTRY (inferior);

/* Create an empty inferior list, or empty the existing one.  */
extern void init_inferior_list (void);

/* Add an inferior to the inferior list, print a message that a new
   inferior is found, and return the pointer to the new inferior.
   Caller may use this pointer to initialize the private inferior
   data.  */
extern struct inferior *add_inferior (int pid);

/* Same as add_inferior, but don't print new inferior notifications to
   the CLI.  */
extern struct inferior *add_inferior_silent (int pid);

/* Delete an existing inferior list entry, due to inferior exit.  */
extern void delete_inferior (int pid);

extern void delete_inferior_1 (struct inferior *todel, int silent);

/* Same as delete_inferior, but don't print new inferior notifications
   to the CLI.  */
extern void delete_inferior_silent (int pid);

/* Delete an existing inferior list entry, due to inferior detaching.  */
extern void detach_inferior (int pid);

extern void exit_inferior (int pid);

extern void exit_inferior_silent (int pid);

extern void exit_inferior_num_silent (int num);

extern void inferior_appeared (struct inferior *inf, int pid);

/* Get rid of all inferiors.  */
extern void discard_all_inferiors (void);

/* Translate the integer inferior id (GDB's homegrown id, not the system's)
   into a "pid" (which may be overloaded with extra inferior information).  */
extern int gdb_inferior_id_to_pid (int);

/* Translate a target 'pid' into the integer inferior id (GDB's
   homegrown id, not the system's).  */
extern int pid_to_gdb_inferior_id (int pid);

/* Boolean test for an already-known pid.  */
extern int in_inferior_list (int pid);

/* Boolean test for an already-known inferior id (GDB's homegrown id,
   not the system's).  */
extern int valid_gdb_inferior_id (int num);

/* Search function to lookup an inferior by target 'pid'.  */
extern struct inferior *find_inferior_pid (int pid);

/* Search function to lookup an inferior by GDB 'num'.  */
extern struct inferior *find_inferior_id (int num);

/* Find an inferior bound to PSPACE.  */
extern struct inferior *
  find_inferior_for_program_space (struct program_space *pspace);

/* Inferior iterator function.

   Calls a callback function once for each inferior, so long as the
   callback function returns false.  If the callback function returns
   true, the iteration will end and the current inferior will be
   returned.  This can be useful for implementing a search for a
   inferior with arbitrary attributes, or for applying some operation
   to every inferior.

   It is safe to delete the iterated inferior from the callback.  */
extern struct inferior *iterate_over_inferiors (int (*) (struct inferior *,
							 void *),
						void *);

/* Returns true if the inferior list is not empty.  */
extern int have_inferiors (void);

/* Returns true if there are any live inferiors in the inferior list
   (not cores, not executables, real live processes).  */
extern int have_live_inferiors (void);

/* Return a pointer to the current inferior.  It is an error to call
   this if there is no current inferior.  */
extern struct inferior *current_inferior (void);

extern void set_current_inferior (struct inferior *);

extern struct cleanup *save_current_inferior (void);

/* Traverse all inferiors.  */

#define ALL_INFERIORS(I) \
  for ((I) = inferior_list; (I); (I) = (I)->next)

extern struct inferior *inferior_list;

/* Prune away automatically added inferiors that aren't required
   anymore.  */
extern void prune_inferiors (void);

extern int number_of_inferiors (void);

extern struct inferior *add_inferior_with_spaces (void);

extern void update_observer_mode (void);

extern void update_signals_program_target (void);

extern void signal_catch_update (const unsigned int *);

/* In some circumstances we allow a command to specify a numeric
   signal.  The idea is to keep these circumstances limited so that
   users (and scripts) develop portable habits.  For comparison,
   POSIX.2 `kill' requires that 1,2,3,6,9,14, and 15 work (and using a
   numeric signal at all is obsolescent.  We are slightly more lenient
   and allow 1-15 which should match host signal numbers on most
   systems.  Use of symbolic signal names is strongly encouraged.  */

enum gdb_signal gdb_signal_from_command (int num);

#endif /* !defined (INFERIOR_H) */
