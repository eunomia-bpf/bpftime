/* Interface between GDB and target environments, including files and processes

   Copyright (C) 1990-2013 Free Software Foundation, Inc.

   Contributed by Cygnus Support.  Written by John Gilmore.

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

#if !defined (TARGET_H)
#define TARGET_H

struct objfile;
struct ui_file;
struct mem_attrib;
struct target_ops;
struct bp_location;
struct bp_target_info;
struct regcache;
struct target_section_table;
struct trace_state_variable;
struct trace_status;
struct uploaded_tsv;
struct uploaded_tp;
struct static_tracepoint_marker;
struct traceframe_info;
struct expression;

/* This include file defines the interface between the main part
   of the debugger, and the part which is target-specific, or
   specific to the communications interface between us and the
   target.

   A TARGET is an interface between the debugger and a particular
   kind of file or process.  Targets can be STACKED in STRATA,
   so that more than one target can potentially respond to a request.
   In particular, memory accesses will walk down the stack of targets
   until they find a target that is interested in handling that particular
   address.  STRATA are artificial boundaries on the stack, within
   which particular kinds of targets live.  Strata exist so that
   people don't get confused by pushing e.g. a process target and then
   a file target, and wondering why they can't see the current values
   of variables any more (the file target is handling them and they
   never get to the process target).  So when you push a file target,
   it goes into the file stratum, which is always below the process
   stratum.  */

#include "bfd.h"
#include "symtab.h"
#include "memattr.h"
#include "vec.h"
#include "gdb_signals.h"
#include "btrace.h"

enum strata
  {
    dummy_stratum,		/* The lowest of the low */
    file_stratum,		/* Executable files, etc */
    process_stratum,		/* Executing processes or core dump files */
    thread_stratum,		/* Executing threads */
    record_stratum,		/* Support record debugging */
    arch_stratum		/* Architecture overrides */
  };

enum thread_control_capabilities
  {
    tc_none = 0,		/* Default: can't control thread execution.  */
    tc_schedlock = 1,		/* Can lock the thread scheduler.  */
  };

/* Stuff for target_wait.  */

/* Generally, what has the program done?  */
enum target_waitkind
  {
    /* The program has exited.  The exit status is in value.integer.  */
    TARGET_WAITKIND_EXITED,

    /* The program has stopped with a signal.  Which signal is in
       value.sig.  */
    TARGET_WAITKIND_STOPPED,

    /* The program has terminated with a signal.  Which signal is in
       value.sig.  */
    TARGET_WAITKIND_SIGNALLED,

    /* The program is letting us know that it dynamically loaded something
       (e.g. it called load(2) on AIX).  */
    TARGET_WAITKIND_LOADED,

    /* The program has forked.  A "related" process' PTID is in
       value.related_pid.  I.e., if the child forks, value.related_pid
       is the parent's ID.  */

    TARGET_WAITKIND_FORKED,

    /* The program has vforked.  A "related" process's PTID is in
       value.related_pid.  */

    TARGET_WAITKIND_VFORKED,

    /* The program has exec'ed a new executable file.  The new file's
       pathname is pointed to by value.execd_pathname.  */

    TARGET_WAITKIND_EXECD,

    /* The program had previously vforked, and now the child is done
       with the shared memory region, because it exec'ed or exited.
       Note that the event is reported to the vfork parent.  This is
       only used if GDB did not stay attached to the vfork child,
       otherwise, a TARGET_WAITKIND_EXECD or
       TARGET_WAITKIND_EXIT|SIGNALLED event associated with the child
       has the same effect.  */
    TARGET_WAITKIND_VFORK_DONE,

    /* The program has entered or returned from a system call.  On
       HP-UX, this is used in the hardware watchpoint implementation.
       The syscall's unique integer ID number is in value.syscall_id.  */

    TARGET_WAITKIND_SYSCALL_ENTRY,
    TARGET_WAITKIND_SYSCALL_RETURN,

    /* Nothing happened, but we stopped anyway.  This perhaps should be handled
       within target_wait, but I'm not sure target_wait should be resuming the
       inferior.  */
    TARGET_WAITKIND_SPURIOUS,

    /* An event has occured, but we should wait again.
       Remote_async_wait() returns this when there is an event
       on the inferior, but the rest of the world is not interested in
       it.  The inferior has not stopped, but has just sent some output
       to the console, for instance.  In this case, we want to go back
       to the event loop and wait there for another event from the
       inferior, rather than being stuck in the remote_async_wait()
       function. sThis way the event loop is responsive to other events,
       like for instance the user typing.  */
    TARGET_WAITKIND_IGNORE,

    /* The target has run out of history information,
       and cannot run backward any further.  */
    TARGET_WAITKIND_NO_HISTORY,

    /* There are no resumed children left in the program.  */
    TARGET_WAITKIND_NO_RESUMED
  };

struct target_waitstatus
  {
    enum target_waitkind kind;

    /* Forked child pid, execd pathname, exit status, signal number or
       syscall number.  */
    union
      {
	int integer;
	enum gdb_signal sig;
	ptid_t related_pid;
	char *execd_pathname;
	int syscall_number;
      }
    value;
  };

/* Options that can be passed to target_wait.  */

/* Return immediately if there's no event already queued.  If this
   options is not requested, target_wait blocks waiting for an
   event.  */
#define TARGET_WNOHANG 1

/* The structure below stores information about a system call.
   It is basically used in the "catch syscall" command, and in
   every function that gives information about a system call.
   
   It's also good to mention that its fields represent everything
   that we currently know about a syscall in GDB.  */
struct syscall
  {
    /* The syscall number.  */
    int number;

    /* The syscall name.  */
    const char *name;
  };

/* Return a pretty printed form of target_waitstatus.
   Space for the result is malloc'd, caller must free.  */
extern char *target_waitstatus_to_string (const struct target_waitstatus *);

/* Return a pretty printed form of TARGET_OPTIONS.
   Space for the result is malloc'd, caller must free.  */
extern char *target_options_to_string (int target_options);

/* Possible types of events that the inferior handler will have to
   deal with.  */
enum inferior_event_type
  {
    /* Process a normal inferior event which will result in target_wait
       being called.  */
    INF_REG_EVENT,
    /* We are called because a timer went off.  */
    INF_TIMER,
    /* We are called to do stuff after the inferior stops.  */
    INF_EXEC_COMPLETE,
    /* We are called to do some stuff after the inferior stops, but we
       are expected to reenter the proceed() and
       handle_inferior_event() functions.  This is used only in case of
       'step n' like commands.  */
    INF_EXEC_CONTINUE
  };

/* Target objects which can be transfered using target_read,
   target_write, et cetera.  */

enum target_object
{
  /* AVR target specific transfer.  See "avr-tdep.c" and "remote.c".  */
  TARGET_OBJECT_AVR,
  /* SPU target specific transfer.  See "spu-tdep.c".  */
  TARGET_OBJECT_SPU,
  /* Transfer up-to LEN bytes of memory starting at OFFSET.  */
  TARGET_OBJECT_MEMORY,
  /* Memory, avoiding GDB's data cache and trusting the executable.
     Target implementations of to_xfer_partial never need to handle
     this object, and most callers should not use it.  */
  TARGET_OBJECT_RAW_MEMORY,
  /* Memory known to be part of the target's stack.  This is cached even
     if it is not in a region marked as such, since it is known to be
     "normal" RAM.  */
  TARGET_OBJECT_STACK_MEMORY,
  /* Kernel Unwind Table.  See "ia64-tdep.c".  */
  TARGET_OBJECT_UNWIND_TABLE,
  /* Transfer auxilliary vector.  */
  TARGET_OBJECT_AUXV,
  /* StackGhost cookie.  See "sparc-tdep.c".  */
  TARGET_OBJECT_WCOOKIE,
  /* Target memory map in XML format.  */
  TARGET_OBJECT_MEMORY_MAP,
  /* Flash memory.  This object can be used to write contents to
     a previously erased flash memory.  Using it without erasing
     flash can have unexpected results.  Addresses are physical
     address on target, and not relative to flash start.  */
  TARGET_OBJECT_FLASH,
  /* Available target-specific features, e.g. registers and coprocessors.
     See "target-descriptions.c".  ANNEX should never be empty.  */
  TARGET_OBJECT_AVAILABLE_FEATURES,
  /* Currently loaded libraries, in XML format.  */
  TARGET_OBJECT_LIBRARIES,
  /* Currently loaded libraries specific for SVR4 systems, in XML format.  */
  TARGET_OBJECT_LIBRARIES_SVR4,
  /* Get OS specific data.  The ANNEX specifies the type (running
     processes, etc.).  The data being transfered is expected to follow
     the DTD specified in features/osdata.dtd.  */
  TARGET_OBJECT_OSDATA,
  /* Extra signal info.  Usually the contents of `siginfo_t' on unix
     platforms.  */
  TARGET_OBJECT_SIGNAL_INFO,
  /* The list of threads that are being debugged.  */
  TARGET_OBJECT_THREADS,
  /* Collected static trace data.  */
  TARGET_OBJECT_STATIC_TRACE_DATA,
  /* The HP-UX registers (those that can be obtained or modified by using
     the TT_LWP_RUREGS/TT_LWP_WUREGS ttrace requests).  */
  TARGET_OBJECT_HPUX_UREGS,
  /* The HP-UX shared library linkage pointer.  ANNEX should be a string
     image of the code address whose linkage pointer we are looking for.

     The size of the data transfered is always 8 bytes (the size of an
     address on ia64).  */
  TARGET_OBJECT_HPUX_SOLIB_GOT,
  /* Traceframe info, in XML format.  */
  TARGET_OBJECT_TRACEFRAME_INFO,
  /* Load maps for FDPIC systems.  */
  TARGET_OBJECT_FDPIC,
  /* Darwin dynamic linker info data.  */
  TARGET_OBJECT_DARWIN_DYLD_INFO,
  /* OpenVMS Unwind Information Block.  */
  TARGET_OBJECT_OPENVMS_UIB,
  /* Branch trace data, in XML format.  */
  TARGET_OBJECT_BTRACE
  /* Possible future objects: TARGET_OBJECT_FILE, ...  */
};

/* Enumeration of the kinds of traceframe searches that a target may
   be able to perform.  */

enum trace_find_type
  {
    tfind_number,
    tfind_pc,
    tfind_tp,
    tfind_range,
    tfind_outside,
  };

typedef struct static_tracepoint_marker *static_tracepoint_marker_p;
DEF_VEC_P(static_tracepoint_marker_p);

/* Request that OPS transfer up to LEN 8-bit bytes of the target's
   OBJECT.  The OFFSET, for a seekable object, specifies the
   starting point.  The ANNEX can be used to provide additional
   data-specific information to the target.

   Return the number of bytes actually transfered, or -1 if the
   transfer is not supported or otherwise fails.  Return of a positive
   value less than LEN indicates that no further transfer is possible.
   Unlike the raw to_xfer_partial interface, callers of these
   functions do not need to retry partial transfers.  */

extern LONGEST target_read (struct target_ops *ops,
			    enum target_object object,
			    const char *annex, gdb_byte *buf,
			    ULONGEST offset, LONGEST len);

struct memory_read_result
  {
    /* First address that was read.  */
    ULONGEST begin;
    /* Past-the-end address.  */
    ULONGEST end;
    /* The data.  */
    gdb_byte *data;
};
typedef struct memory_read_result memory_read_result_s;
DEF_VEC_O(memory_read_result_s);

extern void free_memory_read_result_vector (void *);

extern VEC(memory_read_result_s)* read_memory_robust (struct target_ops *ops,
						      ULONGEST offset,
						      LONGEST len);
  
extern LONGEST target_write (struct target_ops *ops,
			     enum target_object object,
			     const char *annex, const gdb_byte *buf,
			     ULONGEST offset, LONGEST len);

/* Similar to target_write, except that it also calls PROGRESS with
   the number of bytes written and the opaque BATON after every
   successful partial write (and before the first write).  This is
   useful for progress reporting and user interaction while writing
   data.  To abort the transfer, the progress callback can throw an
   exception.  */

LONGEST target_write_with_progress (struct target_ops *ops,
				    enum target_object object,
				    const char *annex, const gdb_byte *buf,
				    ULONGEST offset, LONGEST len,
				    void (*progress) (ULONGEST, void *),
				    void *baton);

/* Wrapper to perform a full read of unknown size.  OBJECT/ANNEX will
   be read using OPS.  The return value will be -1 if the transfer
   fails or is not supported; 0 if the object is empty; or the length
   of the object otherwise.  If a positive value is returned, a
   sufficiently large buffer will be allocated using xmalloc and
   returned in *BUF_P containing the contents of the object.

   This method should be used for objects sufficiently small to store
   in a single xmalloc'd buffer, when no fixed bound on the object's
   size is known in advance.  Don't try to read TARGET_OBJECT_MEMORY
   through this function.  */

extern LONGEST target_read_alloc (struct target_ops *ops,
				  enum target_object object,
				  const char *annex, gdb_byte **buf_p);

/* Read OBJECT/ANNEX using OPS.  The result is NUL-terminated and
   returned as a string, allocated using xmalloc.  If an error occurs
   or the transfer is unsupported, NULL is returned.  Empty objects
   are returned as allocated but empty strings.  A warning is issued
   if the result contains any embedded NUL bytes.  */

extern char *target_read_stralloc (struct target_ops *ops,
				   enum target_object object,
				   const char *annex);

/* Wrappers to target read/write that perform memory transfers.  They
   throw an error if the memory transfer fails.

   NOTE: cagney/2003-10-23: The naming schema is lifted from
   "frame.h".  The parameter order is lifted from get_frame_memory,
   which in turn lifted it from read_memory.  */

extern void get_target_memory (struct target_ops *ops, CORE_ADDR addr,
			       gdb_byte *buf, LONGEST len);
extern ULONGEST get_target_memory_unsigned (struct target_ops *ops,
					    CORE_ADDR addr, int len,
					    enum bfd_endian byte_order);

struct thread_info;		/* fwd decl for parameter list below: */

struct target_ops
  {
    struct target_ops *beneath;	/* To the target under this one.  */
    char *to_shortname;		/* Name this target type */
    char *to_longname;		/* Name for printing */
    char *to_doc;		/* Documentation.  Does not include trailing
				   newline, and starts with a one-line descrip-
				   tion (probably similar to to_longname).  */
    /* Per-target scratch pad.  */
    void *to_data;
    /* The open routine takes the rest of the parameters from the
       command, and (if successful) pushes a new target onto the
       stack.  Targets should supply this routine, if only to provide
       an error message.  */
    void (*to_open) (char *, int);
    /* Old targets with a static target vector provide "to_close".
       New re-entrant targets provide "to_xclose" and that is expected
       to xfree everything (including the "struct target_ops").  */
    void (*to_xclose) (struct target_ops *targ, int quitting);
    void (*to_close) (int);
    void (*to_attach) (struct target_ops *ops, char *, int);
    void (*to_post_attach) (int);
    void (*to_detach) (struct target_ops *ops, char *, int);
    void (*to_disconnect) (struct target_ops *, char *, int);
    void (*to_resume) (struct target_ops *, ptid_t, int, enum gdb_signal);
    ptid_t (*to_wait) (struct target_ops *,
		       ptid_t, struct target_waitstatus *, int);
    void (*to_fetch_registers) (struct target_ops *, struct regcache *, int);
    void (*to_store_registers) (struct target_ops *, struct regcache *, int);
    void (*to_prepare_to_store) (struct regcache *);

    /* Transfer LEN bytes of memory between GDB address MYADDR and
       target address MEMADDR.  If WRITE, transfer them to the target, else
       transfer them from the target.  TARGET is the target from which we
       get this function.

       Return value, N, is one of the following:

       0 means that we can't handle this.  If errno has been set, it is the
       error which prevented us from doing it (FIXME: What about bfd_error?).

       positive (call it N) means that we have transferred N bytes
       starting at MEMADDR.  We might be able to handle more bytes
       beyond this length, but no promises.

       negative (call its absolute value N) means that we cannot
       transfer right at MEMADDR, but we could transfer at least
       something at MEMADDR + N.

       NOTE: cagney/2004-10-01: This has been entirely superseeded by
       to_xfer_partial and inferior inheritance.  */

    int (*deprecated_xfer_memory) (CORE_ADDR memaddr, gdb_byte *myaddr,
				   int len, int write,
				   struct mem_attrib *attrib,
				   struct target_ops *target);

    void (*to_files_info) (struct target_ops *);
    int (*to_insert_breakpoint) (struct gdbarch *, struct bp_target_info *);
    int (*to_remove_breakpoint) (struct gdbarch *, struct bp_target_info *);
    int (*to_can_use_hw_breakpoint) (int, int, int);
    int (*to_ranged_break_num_registers) (struct target_ops *);
    int (*to_insert_hw_breakpoint) (struct gdbarch *, struct bp_target_info *);
    int (*to_remove_hw_breakpoint) (struct gdbarch *, struct bp_target_info *);

    /* Documentation of what the two routines below are expected to do is
       provided with the corresponding target_* macros.  */
    int (*to_remove_watchpoint) (CORE_ADDR, int, int, struct expression *);
    int (*to_insert_watchpoint) (CORE_ADDR, int, int, struct expression *);

    int (*to_insert_mask_watchpoint) (struct target_ops *,
				      CORE_ADDR, CORE_ADDR, int);
    int (*to_remove_mask_watchpoint) (struct target_ops *,
				      CORE_ADDR, CORE_ADDR, int);
    int (*to_stopped_by_watchpoint) (void);
    int to_have_steppable_watchpoint;
    int to_have_continuable_watchpoint;
    int (*to_stopped_data_address) (struct target_ops *, CORE_ADDR *);
    int (*to_watchpoint_addr_within_range) (struct target_ops *,
					    CORE_ADDR, CORE_ADDR, int);

    /* Documentation of this routine is provided with the corresponding
       target_* macro.  */
    int (*to_region_ok_for_hw_watchpoint) (CORE_ADDR, int);

    int (*to_can_accel_watchpoint_condition) (CORE_ADDR, int, int,
					      struct expression *);
    int (*to_masked_watch_num_registers) (struct target_ops *,
					  CORE_ADDR, CORE_ADDR);
    void (*to_terminal_init) (void);
    void (*to_terminal_inferior) (void);
    void (*to_terminal_ours_for_output) (void);
    void (*to_terminal_ours) (void);
    void (*to_terminal_save_ours) (void);
    void (*to_terminal_info) (char *, int);
    void (*to_kill) (struct target_ops *);
    void (*to_load) (char *, int);
    void (*to_create_inferior) (struct target_ops *, 
				char *, char *, char **, int);
    void (*to_post_startup_inferior) (ptid_t);
    int (*to_insert_fork_catchpoint) (int);
    int (*to_remove_fork_catchpoint) (int);
    int (*to_insert_vfork_catchpoint) (int);
    int (*to_remove_vfork_catchpoint) (int);
    int (*to_follow_fork) (struct target_ops *, int);
    int (*to_insert_exec_catchpoint) (int);
    int (*to_remove_exec_catchpoint) (int);
    int (*to_set_syscall_catchpoint) (int, int, int, int, int *);
    int (*to_has_exited) (int, int, int *);
    void (*to_mourn_inferior) (struct target_ops *);
    int (*to_can_run) (void);

    /* Documentation of this routine is provided with the corresponding
       target_* macro.  */
    void (*to_pass_signals) (int, unsigned char *);

    /* Documentation of this routine is provided with the
       corresponding target_* function.  */
    void (*to_program_signals) (int, unsigned char *);

    int (*to_thread_alive) (struct target_ops *, ptid_t ptid);
    void (*to_find_new_threads) (struct target_ops *);
    char *(*to_pid_to_str) (struct target_ops *, ptid_t);
    char *(*to_extra_thread_info) (struct thread_info *);
    char *(*to_thread_name) (struct thread_info *);
    void (*to_stop) (ptid_t);
    void (*to_rcmd) (char *command, struct ui_file *output);
    char *(*to_pid_to_exec_file) (int pid);
    void (*to_log_command) (const char *);
    struct target_section_table *(*to_get_section_table) (struct target_ops *);
    enum strata to_stratum;
    int (*to_has_all_memory) (struct target_ops *);
    int (*to_has_memory) (struct target_ops *);
    int (*to_has_stack) (struct target_ops *);
    int (*to_has_registers) (struct target_ops *);
    int (*to_has_execution) (struct target_ops *, ptid_t);
    int to_has_thread_control;	/* control thread execution */
    int to_attach_no_wait;
    /* ASYNC target controls */
    int (*to_can_async_p) (void);
    int (*to_is_async_p) (void);
    void (*to_async) (void (*) (enum inferior_event_type, void *), void *);
    int (*to_supports_non_stop) (void);
    /* find_memory_regions support method for gcore */
    int (*to_find_memory_regions) (find_memory_region_ftype func, void *data);
    /* make_corefile_notes support method for gcore */
    char * (*to_make_corefile_notes) (bfd *, int *);
    /* get_bookmark support method for bookmarks */
    gdb_byte * (*to_get_bookmark) (char *, int);
    /* goto_bookmark support method for bookmarks */
    void (*to_goto_bookmark) (gdb_byte *, int);
    /* Return the thread-local address at OFFSET in the
       thread-local storage for the thread PTID and the shared library
       or executable file given by OBJFILE.  If that block of
       thread-local storage hasn't been allocated yet, this function
       may return an error.  */
    CORE_ADDR (*to_get_thread_local_address) (struct target_ops *ops,
					      ptid_t ptid,
					      CORE_ADDR load_module_addr,
					      CORE_ADDR offset);

    /* Request that OPS transfer up to LEN 8-bit bytes of the target's
       OBJECT.  The OFFSET, for a seekable object, specifies the
       starting point.  The ANNEX can be used to provide additional
       data-specific information to the target.

       Return the number of bytes actually transfered, zero when no
       further transfer is possible, and -1 when the transfer is not
       supported.  Return of a positive value smaller than LEN does
       not indicate the end of the object, only the end of the
       transfer; higher level code should continue transferring if
       desired.  This is handled in target.c.

       The interface does not support a "retry" mechanism.  Instead it
       assumes that at least one byte will be transfered on each
       successful call.

       NOTE: cagney/2003-10-17: The current interface can lead to
       fragmented transfers.  Lower target levels should not implement
       hacks, such as enlarging the transfer, in an attempt to
       compensate for this.  Instead, the target stack should be
       extended so that it implements supply/collect methods and a
       look-aside object cache.  With that available, the lowest
       target can safely and freely "push" data up the stack.

       See target_read and target_write for more information.  One,
       and only one, of readbuf or writebuf must be non-NULL.  */

    LONGEST (*to_xfer_partial) (struct target_ops *ops,
				enum target_object object, const char *annex,
				gdb_byte *readbuf, const gdb_byte *writebuf,
				ULONGEST offset, LONGEST len);

    /* Returns the memory map for the target.  A return value of NULL
       means that no memory map is available.  If a memory address
       does not fall within any returned regions, it's assumed to be
       RAM.  The returned memory regions should not overlap.

       The order of regions does not matter; target_memory_map will
       sort regions by starting address.  For that reason, this
       function should not be called directly except via
       target_memory_map.

       This method should not cache data; if the memory map could
       change unexpectedly, it should be invalidated, and higher
       layers will re-fetch it.  */
    VEC(mem_region_s) *(*to_memory_map) (struct target_ops *);

    /* Erases the region of flash memory starting at ADDRESS, of
       length LENGTH.

       Precondition: both ADDRESS and ADDRESS+LENGTH should be aligned
       on flash block boundaries, as reported by 'to_memory_map'.  */
    void (*to_flash_erase) (struct target_ops *,
                           ULONGEST address, LONGEST length);

    /* Finishes a flash memory write sequence.  After this operation
       all flash memory should be available for writing and the result
       of reading from areas written by 'to_flash_write' should be
       equal to what was written.  */
    void (*to_flash_done) (struct target_ops *);

    /* Describe the architecture-specific features of this target.
       Returns the description found, or NULL if no description
       was available.  */
    const struct target_desc *(*to_read_description) (struct target_ops *ops);

    /* Build the PTID of the thread on which a given task is running,
       based on LWP and THREAD.  These values are extracted from the
       task Private_Data section of the Ada Task Control Block, and
       their interpretation depends on the target.  */
    ptid_t (*to_get_ada_task_ptid) (long lwp, long thread);

    /* Read one auxv entry from *READPTR, not reading locations >= ENDPTR.
       Return 0 if *READPTR is already at the end of the buffer.
       Return -1 if there is insufficient buffer for a whole entry.
       Return 1 if an entry was read into *TYPEP and *VALP.  */
    int (*to_auxv_parse) (struct target_ops *ops, gdb_byte **readptr,
                         gdb_byte *endptr, CORE_ADDR *typep, CORE_ADDR *valp);

    /* Search SEARCH_SPACE_LEN bytes beginning at START_ADDR for the
       sequence of bytes in PATTERN with length PATTERN_LEN.

       The result is 1 if found, 0 if not found, and -1 if there was an error
       requiring halting of the search (e.g. memory read error).
       If the pattern is found the address is recorded in FOUND_ADDRP.  */
    int (*to_search_memory) (struct target_ops *ops,
			     CORE_ADDR start_addr, ULONGEST search_space_len,
			     const gdb_byte *pattern, ULONGEST pattern_len,
			     CORE_ADDR *found_addrp);

    /* Can target execute in reverse?  */
    int (*to_can_execute_reverse) (void);

    /* The direction the target is currently executing.  Must be
       implemented on targets that support reverse execution and async
       mode.  The default simply returns forward execution.  */
    enum exec_direction_kind (*to_execution_direction) (void);

    /* Does this target support debugging multiple processes
       simultaneously?  */
    int (*to_supports_multi_process) (void);

    /* Does this target support enabling and disabling tracepoints while a trace
       experiment is running?  */
    int (*to_supports_enable_disable_tracepoint) (void);

    /* Does this target support disabling address space randomization?  */
    int (*to_supports_disable_randomization) (void);

    /* Does this target support the tracenz bytecode for string collection?  */
    int (*to_supports_string_tracing) (void);

    /* Does this target support evaluation of breakpoint conditions on its
       end?  */
    int (*to_supports_evaluation_of_breakpoint_conditions) (void);

    /* Does this target support evaluation of breakpoint commands on its
       end?  */
    int (*to_can_run_breakpoint_commands) (void);

    /* Determine current architecture of thread PTID.

       The target is supposed to determine the architecture of the code where
       the target is currently stopped at (on Cell, if a target is in spu_run,
       to_thread_architecture would return SPU, otherwise PPC32 or PPC64).
       This is architecture used to perform decr_pc_after_break adjustment,
       and also determines the frame architecture of the innermost frame.
       ptrace operations need to operate according to target_gdbarch ().

       The default implementation always returns target_gdbarch ().  */
    struct gdbarch *(*to_thread_architecture) (struct target_ops *, ptid_t);

    /* Determine current address space of thread PTID.

       The default implementation always returns the inferior's
       address space.  */
    struct address_space *(*to_thread_address_space) (struct target_ops *,
						      ptid_t);

    /* Target file operations.  */

    /* Open FILENAME on the target, using FLAGS and MODE.  Return a
       target file descriptor, or -1 if an error occurs (and set
       *TARGET_ERRNO).  */
    int (*to_fileio_open) (const char *filename, int flags, int mode,
			   int *target_errno);

    /* Write up to LEN bytes from WRITE_BUF to FD on the target.
       Return the number of bytes written, or -1 if an error occurs
       (and set *TARGET_ERRNO).  */
    int (*to_fileio_pwrite) (int fd, const gdb_byte *write_buf, int len,
			     ULONGEST offset, int *target_errno);

    /* Read up to LEN bytes FD on the target into READ_BUF.
       Return the number of bytes read, or -1 if an error occurs
       (and set *TARGET_ERRNO).  */
    int (*to_fileio_pread) (int fd, gdb_byte *read_buf, int len,
			    ULONGEST offset, int *target_errno);

    /* Close FD on the target.  Return 0, or -1 if an error occurs
       (and set *TARGET_ERRNO).  */
    int (*to_fileio_close) (int fd, int *target_errno);

    /* Unlink FILENAME on the target.  Return 0, or -1 if an error
       occurs (and set *TARGET_ERRNO).  */
    int (*to_fileio_unlink) (const char *filename, int *target_errno);

    /* Read value of symbolic link FILENAME on the target.  Return a
       null-terminated string allocated via xmalloc, or NULL if an error
       occurs (and set *TARGET_ERRNO).  */
    char *(*to_fileio_readlink) (const char *filename, int *target_errno);


    /* Implement the "info proc" command.  */
    void (*to_info_proc) (struct target_ops *, char *, enum info_proc_what);

    /* Tracepoint-related operations.  */

    /* Prepare the target for a tracing run.  */
    void (*to_trace_init) (void);

    /* Send full details of a tracepoint location to the target.  */
    void (*to_download_tracepoint) (struct bp_location *location);

    /* Is the target able to download tracepoint locations in current
       state?  */
    int (*to_can_download_tracepoint) (void);

    /* Send full details of a trace state variable to the target.  */
    void (*to_download_trace_state_variable) (struct trace_state_variable *tsv);

    /* Enable a tracepoint on the target.  */
    void (*to_enable_tracepoint) (struct bp_location *location);

    /* Disable a tracepoint on the target.  */
    void (*to_disable_tracepoint) (struct bp_location *location);

    /* Inform the target info of memory regions that are readonly
       (such as text sections), and so it should return data from
       those rather than look in the trace buffer.  */
    void (*to_trace_set_readonly_regions) (void);

    /* Start a trace run.  */
    void (*to_trace_start) (void);

    /* Get the current status of a tracing run.  */
    int (*to_get_trace_status) (struct trace_status *ts);

    void (*to_get_tracepoint_status) (struct breakpoint *tp,
				      struct uploaded_tp *utp);

    /* Stop a trace run.  */
    void (*to_trace_stop) (void);

   /* Ask the target to find a trace frame of the given type TYPE,
      using NUM, ADDR1, and ADDR2 as search parameters.  Returns the
      number of the trace frame, and also the tracepoint number at
      TPP.  If no trace frame matches, return -1.  May throw if the
      operation fails.  */
    int (*to_trace_find) (enum trace_find_type type, int num,
			  ULONGEST addr1, ULONGEST addr2, int *tpp);

    /* Get the value of the trace state variable number TSV, returning
       1 if the value is known and writing the value itself into the
       location pointed to by VAL, else returning 0.  */
    int (*to_get_trace_state_variable_value) (int tsv, LONGEST *val);

    int (*to_save_trace_data) (const char *filename);

    int (*to_upload_tracepoints) (struct uploaded_tp **utpp);

    int (*to_upload_trace_state_variables) (struct uploaded_tsv **utsvp);

    LONGEST (*to_get_raw_trace_data) (gdb_byte *buf,
				      ULONGEST offset, LONGEST len);

    /* Get the minimum length of instruction on which a fast tracepoint
       may be set on the target.  If this operation is unsupported,
       return -1.  If for some reason the minimum length cannot be
       determined, return 0.  */
    int (*to_get_min_fast_tracepoint_insn_len) (void);

    /* Set the target's tracing behavior in response to unexpected
       disconnection - set VAL to 1 to keep tracing, 0 to stop.  */
    void (*to_set_disconnected_tracing) (int val);
    void (*to_set_circular_trace_buffer) (int val);
    /* Set the size of trace buffer in the target.  */
    void (*to_set_trace_buffer_size) (LONGEST val);

    /* Add/change textual notes about the trace run, returning 1 if
       successful, 0 otherwise.  */
    int (*to_set_trace_notes) (char *user, char *notes, char* stopnotes);

    /* Return the processor core that thread PTID was last seen on.
       This information is updated only when:
       - update_thread_list is called
       - thread stops
       If the core cannot be determined -- either for the specified
       thread, or right now, or in this debug session, or for this
       target -- return -1.  */
    int (*to_core_of_thread) (struct target_ops *, ptid_t ptid);

    /* Verify that the memory in the [MEMADDR, MEMADDR+SIZE) range
       matches the contents of [DATA,DATA+SIZE).  Returns 1 if there's
       a match, 0 if there's a mismatch, and -1 if an error is
       encountered while reading memory.  */
    int (*to_verify_memory) (struct target_ops *, const gdb_byte *data,
			     CORE_ADDR memaddr, ULONGEST size);

    /* Return the address of the start of the Thread Information Block
       a Windows OS specific feature.  */
    int (*to_get_tib_address) (ptid_t ptid, CORE_ADDR *addr);

    /* Send the new settings of write permission variables.  */
    void (*to_set_permissions) (void);

    /* Look for a static tracepoint marker at ADDR, and fill in MARKER
       with its details.  Return 1 on success, 0 on failure.  */
    int (*to_static_tracepoint_marker_at) (CORE_ADDR,
					   struct static_tracepoint_marker *marker);

    /* Return a vector of all tracepoints markers string id ID, or all
       markers if ID is NULL.  */
    VEC(static_tracepoint_marker_p) *(*to_static_tracepoint_markers_by_strid)
      (const char *id);

    /* Return a traceframe info object describing the current
       traceframe's contents.  This method should not cache data;
       higher layers take care of caching, invalidating, and
       re-fetching when necessary.  */
    struct traceframe_info *(*to_traceframe_info) (void);

    /* Ask the target to use or not to use agent according to USE.  Return 1
       successful, 0 otherwise.  */
    int (*to_use_agent) (int use);

    /* Is the target able to use agent in current state?  */
    int (*to_can_use_agent) (void);

    /* Check whether the target supports branch tracing.  */
    int (*to_supports_btrace) (void);

    /* Enable branch tracing for PTID and allocate a branch trace target
       information struct for reading and for disabling branch trace.  */
    struct btrace_target_info *(*to_enable_btrace) (ptid_t ptid);

    /* Disable branch tracing and deallocate TINFO.  */
    void (*to_disable_btrace) (struct btrace_target_info *tinfo);

    /* Disable branch tracing and deallocate TINFO.  This function is similar
       to to_disable_btrace, except that it is called during teardown and is
       only allowed to perform actions that are safe.  A counter-example would
       be attempting to talk to a remote target.  */
    void (*to_teardown_btrace) (struct btrace_target_info *tinfo);

    /* Read branch trace data.  */
    VEC (btrace_block_s) *(*to_read_btrace) (struct btrace_target_info *,
					     enum btrace_read_type);

    /* Stop trace recording.  */
    void (*to_stop_recording) (void);

    /* Print information about the recording.  */
    void (*to_info_record) (void);

    /* Save the recorded execution trace into a file.  */
    void (*to_save_record) (char *filename);

    /* Delete the recorded execution trace from the current position onwards.  */
    void (*to_delete_record) (void);

    /* Query if the record target is currently replaying.  */
    int (*to_record_is_replaying) (void);

    /* Go to the begin of the execution trace.  */
    void (*to_goto_record_begin) (void);

    /* Go to the end of the execution trace.  */
    void (*to_goto_record_end) (void);

    /* Go to a specific location in the recorded execution trace.  */
    void (*to_goto_record) (ULONGEST insn);

    /* Disassemble SIZE instructions in the recorded execution trace from
       the current position.
       If SIZE < 0, disassemble abs (SIZE) preceding instructions; otherwise,
       disassemble SIZE succeeding instructions.  */
    void (*to_insn_history) (int size, int flags);

    /* Disassemble SIZE instructions in the recorded execution trace around
       FROM.
       If SIZE < 0, disassemble abs (SIZE) instructions before FROM; otherwise,
       disassemble SIZE instructions after FROM.  */
    void (*to_insn_history_from) (ULONGEST from, int size, int flags);

    /* Disassemble a section of the recorded execution trace from instruction
       BEGIN (inclusive) to instruction END (exclusive).  */
    void (*to_insn_history_range) (ULONGEST begin, ULONGEST end, int flags);

    /* Print a function trace of the recorded execution trace.
       If SIZE < 0, print abs (SIZE) preceding functions; otherwise, print SIZE
       succeeding functions.  */
    void (*to_call_history) (int size, int flags);

    /* Print a function trace of the recorded execution trace starting
       at function FROM.
       If SIZE < 0, print abs (SIZE) functions before FROM; otherwise, print
       SIZE functions after FROM.  */
    void (*to_call_history_from) (ULONGEST begin, int size, int flags);

    /* Print a function trace of an execution trace section from function BEGIN
       (inclusive) to function END (exclusive).  */
    void (*to_call_history_range) (ULONGEST begin, ULONGEST end, int flags);

    int to_magic;
    /* Need sub-structure for target machine related rather than comm related?
     */
  };

/* Magic number for checking ops size.  If a struct doesn't end with this
   number, somebody changed the declaration but didn't change all the
   places that initialize one.  */

#define	OPS_MAGIC	3840

/* The ops structure for our "current" target process.  This should
   never be NULL.  If there is no target, it points to the dummy_target.  */

extern struct target_ops current_target;

/* Define easy words for doing these operations on our current target.  */

#define	target_shortname	(current_target.to_shortname)
#define	target_longname		(current_target.to_longname)

/* Does whatever cleanup is required for a target that we are no
   longer going to be calling.  QUITTING indicates that GDB is exiting
   and should not get hung on an error (otherwise it is important to
   perform clean termination, even if it takes a while).  This routine
   is automatically always called after popping the target off the
   target stack - the target's own methods are no longer available
   through the target vector.  Closing file descriptors and freeing all
   memory allocated memory are typical things it should do.  */

void target_close (struct target_ops *targ, int quitting);

/* Attaches to a process on the target side.  Arguments are as passed
   to the `attach' command by the user.  This routine can be called
   when the target is not on the target-stack, if the target_can_run
   routine returns 1; in that case, it must push itself onto the stack.
   Upon exit, the target should be ready for normal operations, and
   should be ready to deliver the status of the process immediately
   (without waiting) to an upcoming target_wait call.  */

void target_attach (char *, int);

/* Some targets don't generate traps when attaching to the inferior,
   or their target_attach implementation takes care of the waiting.
   These targets must set to_attach_no_wait.  */

#define target_attach_no_wait \
     (current_target.to_attach_no_wait)

/* The target_attach operation places a process under debugger control,
   and stops the process.

   This operation provides a target-specific hook that allows the
   necessary bookkeeping to be performed after an attach completes.  */
#define target_post_attach(pid) \
     (*current_target.to_post_attach) (pid)

/* Takes a program previously attached to and detaches it.
   The program may resume execution (some targets do, some don't) and will
   no longer stop on signals, etc.  We better not have left any breakpoints
   in the program or it'll die when it hits one.  ARGS is arguments
   typed by the user (e.g. a signal to send the process).  FROM_TTY
   says whether to be verbose or not.  */

extern void target_detach (char *, int);

/* Disconnect from the current target without resuming it (leaving it
   waiting for a debugger).  */

extern void target_disconnect (char *, int);

/* Resume execution of the target process PTID (or a group of
   threads).  STEP says whether to single-step or to run free; SIGGNAL
   is the signal to be given to the target, or GDB_SIGNAL_0 for no
   signal.  The caller may not pass GDB_SIGNAL_DEFAULT.  A specific
   PTID means `step/resume only this process id'.  A wildcard PTID
   (all threads, or all threads of process) means `step/resume
   INFERIOR_PTID, and let other threads (for which the wildcard PTID
   matches) resume with their 'thread->suspend.stop_signal' signal
   (usually GDB_SIGNAL_0) if it is in "pass" state, or with no signal
   if in "no pass" state.  */

extern void target_resume (ptid_t ptid, int step, enum gdb_signal signal);

/* Wait for process pid to do something.  PTID = -1 to wait for any
   pid to do something.  Return pid of child, or -1 in case of error;
   store status through argument pointer STATUS.  Note that it is
   _NOT_ OK to throw_exception() out of target_wait() without popping
   the debugging target from the stack; GDB isn't prepared to get back
   to the prompt with a debugging target but without the frame cache,
   stop_pc, etc., set up.  OPTIONS is a bitwise OR of TARGET_W*
   options.  */

extern ptid_t target_wait (ptid_t ptid, struct target_waitstatus *status,
			   int options);

/* Fetch at least register REGNO, or all regs if regno == -1.  No result.  */

extern void target_fetch_registers (struct regcache *regcache, int regno);

/* Store at least register REGNO, or all regs if REGNO == -1.
   It can store as many registers as it wants to, so target_prepare_to_store
   must have been previously called.  Calls error() if there are problems.  */

extern void target_store_registers (struct regcache *regcache, int regs);

/* Get ready to modify the registers array.  On machines which store
   individual registers, this doesn't need to do anything.  On machines
   which store all the registers in one fell swoop, this makes sure
   that REGISTERS contains all the registers from the program being
   debugged.  */

#define	target_prepare_to_store(regcache)	\
     (*current_target.to_prepare_to_store) (regcache)

/* Determine current address space of thread PTID.  */

struct address_space *target_thread_address_space (ptid_t);

/* Implement the "info proc" command.  This returns one if the request
   was handled, and zero otherwise.  It can also throw an exception if
   an error was encountered while attempting to handle the
   request.  */

int target_info_proc (char *, enum info_proc_what);

/* Returns true if this target can debug multiple processes
   simultaneously.  */

#define	target_supports_multi_process()	\
     (*current_target.to_supports_multi_process) ()

/* Returns true if this target can disable address space randomization.  */

int target_supports_disable_randomization (void);

/* Returns true if this target can enable and disable tracepoints
   while a trace experiment is running.  */

#define target_supports_enable_disable_tracepoint() \
  (*current_target.to_supports_enable_disable_tracepoint) ()

#define target_supports_string_tracing() \
  (*current_target.to_supports_string_tracing) ()

/* Returns true if this target can handle breakpoint conditions
   on its end.  */

#define target_supports_evaluation_of_breakpoint_conditions() \
  (*current_target.to_supports_evaluation_of_breakpoint_conditions) ()

/* Returns true if this target can handle breakpoint commands
   on its end.  */

#define target_can_run_breakpoint_commands() \
  (*current_target.to_can_run_breakpoint_commands) ()

/* Invalidate all target dcaches.  */
extern void target_dcache_invalidate (void);

extern int target_read_string (CORE_ADDR, char **, int, int *);

extern int target_read_memory (CORE_ADDR memaddr, gdb_byte *myaddr,
			       ssize_t len);

extern int target_read_stack (CORE_ADDR memaddr, gdb_byte *myaddr, ssize_t len);

extern int target_write_memory (CORE_ADDR memaddr, const gdb_byte *myaddr,
				ssize_t len);

extern int target_write_raw_memory (CORE_ADDR memaddr, const gdb_byte *myaddr,
				    ssize_t len);

/* Fetches the target's memory map.  If one is found it is sorted
   and returned, after some consistency checking.  Otherwise, NULL
   is returned.  */
VEC(mem_region_s) *target_memory_map (void);

/* Erase the specified flash region.  */
void target_flash_erase (ULONGEST address, LONGEST length);

/* Finish a sequence of flash operations.  */
void target_flash_done (void);

/* Describes a request for a memory write operation.  */
struct memory_write_request
  {
    /* Begining address that must be written.  */
    ULONGEST begin;
    /* Past-the-end address.  */
    ULONGEST end;
    /* The data to write.  */
    gdb_byte *data;
    /* A callback baton for progress reporting for this request.  */
    void *baton;
  };
typedef struct memory_write_request memory_write_request_s;
DEF_VEC_O(memory_write_request_s);

/* Enumeration specifying different flash preservation behaviour.  */
enum flash_preserve_mode
  {
    flash_preserve,
    flash_discard
  };

/* Write several memory blocks at once.  This version can be more
   efficient than making several calls to target_write_memory, in
   particular because it can optimize accesses to flash memory.

   Moreover, this is currently the only memory access function in gdb
   that supports writing to flash memory, and it should be used for
   all cases where access to flash memory is desirable.

   REQUESTS is the vector (see vec.h) of memory_write_request.
   PRESERVE_FLASH_P indicates what to do with blocks which must be
     erased, but not completely rewritten.
   PROGRESS_CB is a function that will be periodically called to provide
     feedback to user.  It will be called with the baton corresponding
     to the request currently being written.  It may also be called
     with a NULL baton, when preserved flash sectors are being rewritten.

   The function returns 0 on success, and error otherwise.  */
int target_write_memory_blocks (VEC(memory_write_request_s) *requests,
				enum flash_preserve_mode preserve_flash_p,
				void (*progress_cb) (ULONGEST, void *));

/* Print a line about the current target.  */

#define	target_files_info()	\
     (*current_target.to_files_info) (&current_target)

/* Insert a breakpoint at address BP_TGT->placed_address in the target
   machine.  Result is 0 for success, or an errno value.  */

extern int target_insert_breakpoint (struct gdbarch *gdbarch,
				     struct bp_target_info *bp_tgt);

/* Remove a breakpoint at address BP_TGT->placed_address in the target
   machine.  Result is 0 for success, or an errno value.  */

extern int target_remove_breakpoint (struct gdbarch *gdbarch,
				     struct bp_target_info *bp_tgt);

/* Initialize the terminal settings we record for the inferior,
   before we actually run the inferior.  */

#define target_terminal_init() \
     (*current_target.to_terminal_init) ()

/* Put the inferior's terminal settings into effect.
   This is preparation for starting or resuming the inferior.  */

extern void target_terminal_inferior (void);

/* Put some of our terminal settings into effect,
   enough to get proper results from our output,
   but do not change into or out of RAW mode
   so that no input is discarded.

   After doing this, either terminal_ours or terminal_inferior
   should be called to get back to a normal state of affairs.  */

#define target_terminal_ours_for_output() \
     (*current_target.to_terminal_ours_for_output) ()

/* Put our terminal settings into effect.
   First record the inferior's terminal settings
   so they can be restored properly later.  */

#define target_terminal_ours() \
     (*current_target.to_terminal_ours) ()

/* Save our terminal settings.
   This is called from TUI after entering or leaving the curses
   mode.  Since curses modifies our terminal this call is here
   to take this change into account.  */

#define target_terminal_save_ours() \
     (*current_target.to_terminal_save_ours) ()

/* Print useful information about our terminal status, if such a thing
   exists.  */

#define target_terminal_info(arg, from_tty) \
     (*current_target.to_terminal_info) (arg, from_tty)

/* Kill the inferior process.   Make it go away.  */

extern void target_kill (void);

/* Load an executable file into the target process.  This is expected
   to not only bring new code into the target process, but also to
   update GDB's symbol tables to match.

   ARG contains command-line arguments, to be broken down with
   buildargv ().  The first non-switch argument is the filename to
   load, FILE; the second is a number (as parsed by strtoul (..., ...,
   0)), which is an offset to apply to the load addresses of FILE's
   sections.  The target may define switches, or other non-switch
   arguments, as it pleases.  */

extern void target_load (char *arg, int from_tty);

/* Start an inferior process and set inferior_ptid to its pid.
   EXEC_FILE is the file to run.
   ALLARGS is a string containing the arguments to the program.
   ENV is the environment vector to pass.  Errors reported with error().
   On VxWorks and various standalone systems, we ignore exec_file.  */

void target_create_inferior (char *exec_file, char *args,
			     char **env, int from_tty);

/* Some targets (such as ttrace-based HPUX) don't allow us to request
   notification of inferior events such as fork and vork immediately
   after the inferior is created.  (This because of how gdb gets an
   inferior created via invoking a shell to do it.  In such a scenario,
   if the shell init file has commands in it, the shell will fork and
   exec for each of those commands, and we will see each such fork
   event.  Very bad.)

   Such targets will supply an appropriate definition for this function.  */

#define target_post_startup_inferior(ptid) \
     (*current_target.to_post_startup_inferior) (ptid)

/* On some targets, we can catch an inferior fork or vfork event when
   it occurs.  These functions insert/remove an already-created
   catchpoint for such events.  They return  0 for success, 1 if the
   catchpoint type is not supported and -1 for failure.  */

#define target_insert_fork_catchpoint(pid) \
     (*current_target.to_insert_fork_catchpoint) (pid)

#define target_remove_fork_catchpoint(pid) \
     (*current_target.to_remove_fork_catchpoint) (pid)

#define target_insert_vfork_catchpoint(pid) \
     (*current_target.to_insert_vfork_catchpoint) (pid)

#define target_remove_vfork_catchpoint(pid) \
     (*current_target.to_remove_vfork_catchpoint) (pid)

/* If the inferior forks or vforks, this function will be called at
   the next resume in order to perform any bookkeeping and fiddling
   necessary to continue debugging either the parent or child, as
   requested, and releasing the other.  Information about the fork
   or vfork event is available via get_last_target_status ().
   This function returns 1 if the inferior should not be resumed
   (i.e. there is another event pending).  */

int target_follow_fork (int follow_child);

/* On some targets, we can catch an inferior exec event when it
   occurs.  These functions insert/remove an already-created
   catchpoint for such events.  They return  0 for success, 1 if the
   catchpoint type is not supported and -1 for failure.  */

#define target_insert_exec_catchpoint(pid) \
     (*current_target.to_insert_exec_catchpoint) (pid)

#define target_remove_exec_catchpoint(pid) \
     (*current_target.to_remove_exec_catchpoint) (pid)

/* Syscall catch.

   NEEDED is nonzero if any syscall catch (of any kind) is requested.
   If NEEDED is zero, it means the target can disable the mechanism to
   catch system calls because there are no more catchpoints of this type.

   ANY_COUNT is nonzero if a generic (filter-less) syscall catch is
   being requested.  In this case, both TABLE_SIZE and TABLE should
   be ignored.

   TABLE_SIZE is the number of elements in TABLE.  It only matters if
   ANY_COUNT is zero.

   TABLE is an array of ints, indexed by syscall number.  An element in
   this array is nonzero if that syscall should be caught.  This argument
   only matters if ANY_COUNT is zero.

   Return 0 for success, 1 if syscall catchpoints are not supported or -1
   for failure.  */

#define target_set_syscall_catchpoint(pid, needed, any_count, table_size, table) \
     (*current_target.to_set_syscall_catchpoint) (pid, needed, any_count, \
						  table_size, table)

/* Returns TRUE if PID has exited.  And, also sets EXIT_STATUS to the
   exit code of PID, if any.  */

#define target_has_exited(pid,wait_status,exit_status) \
     (*current_target.to_has_exited) (pid,wait_status,exit_status)

/* The debugger has completed a blocking wait() call.  There is now
   some process event that must be processed.  This function should
   be defined by those targets that require the debugger to perform
   cleanup or internal state changes in response to the process event.  */

/* The inferior process has died.  Do what is right.  */

void target_mourn_inferior (void);

/* Does target have enough data to do a run or attach command? */

#define target_can_run(t) \
     ((t)->to_can_run) ()

/* Set list of signals to be handled in the target.

   PASS_SIGNALS is an array of size NSIG, indexed by target signal number
   (enum gdb_signal).  For every signal whose entry in this array is
   non-zero, the target is allowed -but not required- to skip reporting
   arrival of the signal to the GDB core by returning from target_wait,
   and to pass the signal directly to the inferior instead.

   However, if the target is hardware single-stepping a thread that is
   about to receive a signal, it needs to be reported in any case, even
   if mentioned in a previous target_pass_signals call.   */

extern void target_pass_signals (int nsig, unsigned char *pass_signals);

/* Set list of signals the target may pass to the inferior.  This
   directly maps to the "handle SIGNAL pass/nopass" setting.

   PROGRAM_SIGNALS is an array of size NSIG, indexed by target signal
   number (enum gdb_signal).  For every signal whose entry in this
   array is non-zero, the target is allowed to pass the signal to the
   inferior.  Signals not present in the array shall be silently
   discarded.  This does not influence whether to pass signals to the
   inferior as a result of a target_resume call.  This is useful in
   scenarios where the target needs to decide whether to pass or not a
   signal to the inferior without GDB core involvement, such as for
   example, when detaching (as threads may have been suspended with
   pending signals not reported to GDB).  */

extern void target_program_signals (int nsig, unsigned char *program_signals);

/* Check to see if a thread is still alive.  */

extern int target_thread_alive (ptid_t ptid);

/* Query for new threads and add them to the thread list.  */

extern void target_find_new_threads (void);

/* Make target stop in a continuable fashion.  (For instance, under
   Unix, this should act like SIGSTOP).  This function is normally
   used by GUIs to implement a stop button.  */

extern void target_stop (ptid_t ptid);

/* Send the specified COMMAND to the target's monitor
   (shell,interpreter) for execution.  The result of the query is
   placed in OUTBUF.  */

#define target_rcmd(command, outbuf) \
     (*current_target.to_rcmd) (command, outbuf)


/* Does the target include all of memory, or only part of it?  This
   determines whether we look up the target chain for other parts of
   memory if this target can't satisfy a request.  */

extern int target_has_all_memory_1 (void);
#define target_has_all_memory target_has_all_memory_1 ()

/* Does the target include memory?  (Dummy targets don't.)  */

extern int target_has_memory_1 (void);
#define target_has_memory target_has_memory_1 ()

/* Does the target have a stack?  (Exec files don't, VxWorks doesn't, until
   we start a process.)  */

extern int target_has_stack_1 (void);
#define target_has_stack target_has_stack_1 ()

/* Does the target have registers?  (Exec files don't.)  */

extern int target_has_registers_1 (void);
#define target_has_registers target_has_registers_1 ()

/* Does the target have execution?  Can we make it jump (through
   hoops), or pop its stack a few times?  This means that the current
   target is currently executing; for some targets, that's the same as
   whether or not the target is capable of execution, but there are
   also targets which can be current while not executing.  In that
   case this will become true after target_create_inferior or
   target_attach.  */

extern int target_has_execution_1 (ptid_t);

/* Like target_has_execution_1, but always passes inferior_ptid.  */

extern int target_has_execution_current (void);

#define target_has_execution target_has_execution_current ()

/* Default implementations for process_stratum targets.  Return true
   if there's a selected inferior, false otherwise.  */

extern int default_child_has_all_memory (struct target_ops *ops);
extern int default_child_has_memory (struct target_ops *ops);
extern int default_child_has_stack (struct target_ops *ops);
extern int default_child_has_registers (struct target_ops *ops);
extern int default_child_has_execution (struct target_ops *ops,
					ptid_t the_ptid);

/* Can the target support the debugger control of thread execution?
   Can it lock the thread scheduler?  */

#define target_can_lock_scheduler \
     (current_target.to_has_thread_control & tc_schedlock)

/* Should the target enable async mode if it is supported?  Temporary
   cludge until async mode is a strict superset of sync mode.  */
extern int target_async_permitted;

/* Can the target support asynchronous execution?  */
#define target_can_async_p() (current_target.to_can_async_p ())

/* Is the target in asynchronous execution mode?  */
#define target_is_async_p() (current_target.to_is_async_p ())

int target_supports_non_stop (void);

/* Put the target in async mode with the specified callback function.  */
#define target_async(CALLBACK,CONTEXT) \
     (current_target.to_async ((CALLBACK), (CONTEXT)))

#define target_execution_direction() \
  (current_target.to_execution_direction ())

/* Converts a process id to a string.  Usually, the string just contains
   `process xyz', but on some systems it may contain
   `process xyz thread abc'.  */

extern char *target_pid_to_str (ptid_t ptid);

extern char *normal_pid_to_str (ptid_t ptid);

/* Return a short string describing extra information about PID,
   e.g. "sleeping", "runnable", "running on LWP 3".  Null return value
   is okay.  */

#define target_extra_thread_info(TP) \
     (current_target.to_extra_thread_info (TP))

/* Return the thread's name.  A NULL result means that the target
   could not determine this thread's name.  */

extern char *target_thread_name (struct thread_info *);

/* Attempts to find the pathname of the executable file
   that was run to create a specified process.

   The process PID must be stopped when this operation is used.

   If the executable file cannot be determined, NULL is returned.

   Else, a pointer to a character string containing the pathname
   is returned.  This string should be copied into a buffer by
   the client if the string will not be immediately used, or if
   it must persist.  */

#define target_pid_to_exec_file(pid) \
     (current_target.to_pid_to_exec_file) (pid)

/* See the to_thread_architecture description in struct target_ops.  */

#define target_thread_architecture(ptid) \
     (current_target.to_thread_architecture (&current_target, ptid))

/*
 * Iterator function for target memory regions.
 * Calls a callback function once for each memory region 'mapped'
 * in the child process.  Defined as a simple macro rather than
 * as a function macro so that it can be tested for nullity.
 */

#define target_find_memory_regions(FUNC, DATA) \
     (current_target.to_find_memory_regions) (FUNC, DATA)

/*
 * Compose corefile .note section.
 */

#define target_make_corefile_notes(BFD, SIZE_P) \
     (current_target.to_make_corefile_notes) (BFD, SIZE_P)

/* Bookmark interfaces.  */
#define target_get_bookmark(ARGS, FROM_TTY) \
     (current_target.to_get_bookmark) (ARGS, FROM_TTY)

#define target_goto_bookmark(ARG, FROM_TTY) \
     (current_target.to_goto_bookmark) (ARG, FROM_TTY)

/* Hardware watchpoint interfaces.  */

/* Returns non-zero if we were stopped by a hardware watchpoint (memory read or
   write).  Only the INFERIOR_PTID task is being queried.  */

#define target_stopped_by_watchpoint \
   (*current_target.to_stopped_by_watchpoint)

/* Non-zero if we have steppable watchpoints  */

#define target_have_steppable_watchpoint \
   (current_target.to_have_steppable_watchpoint)

/* Non-zero if we have continuable watchpoints  */

#define target_have_continuable_watchpoint \
   (current_target.to_have_continuable_watchpoint)

/* Provide defaults for hardware watchpoint functions.  */

/* If the *_hw_beakpoint functions have not been defined
   elsewhere use the definitions in the target vector.  */

/* Returns non-zero if we can set a hardware watchpoint of type TYPE.  TYPE is
   one of bp_hardware_watchpoint, bp_read_watchpoint, bp_write_watchpoint, or
   bp_hardware_breakpoint.  CNT is the number of such watchpoints used so far
   (including this one?).  OTHERTYPE is who knows what...  */

#define target_can_use_hardware_watchpoint(TYPE,CNT,OTHERTYPE) \
 (*current_target.to_can_use_hw_breakpoint) (TYPE, CNT, OTHERTYPE);

/* Returns the number of debug registers needed to watch the given
   memory region, or zero if not supported.  */

#define target_region_ok_for_hw_watchpoint(addr, len) \
    (*current_target.to_region_ok_for_hw_watchpoint) (addr, len)


/* Set/clear a hardware watchpoint starting at ADDR, for LEN bytes.
   TYPE is 0 for write, 1 for read, and 2 for read/write accesses.
   COND is the expression for its condition, or NULL if there's none.
   Returns 0 for success, 1 if the watchpoint type is not supported,
   -1 for failure.  */

#define	target_insert_watchpoint(addr, len, type, cond) \
     (*current_target.to_insert_watchpoint) (addr, len, type, cond)

#define	target_remove_watchpoint(addr, len, type, cond) \
     (*current_target.to_remove_watchpoint) (addr, len, type, cond)

/* Insert a new masked watchpoint at ADDR using the mask MASK.
   RW may be hw_read for a read watchpoint, hw_write for a write watchpoint
   or hw_access for an access watchpoint.  Returns 0 for success, 1 if
   masked watchpoints are not supported, -1 for failure.  */

extern int target_insert_mask_watchpoint (CORE_ADDR, CORE_ADDR, int);

/* Remove a masked watchpoint at ADDR with the mask MASK.
   RW may be hw_read for a read watchpoint, hw_write for a write watchpoint
   or hw_access for an access watchpoint.  Returns 0 for success, non-zero
   for failure.  */

extern int target_remove_mask_watchpoint (CORE_ADDR, CORE_ADDR, int);

#define target_insert_hw_breakpoint(gdbarch, bp_tgt) \
     (*current_target.to_insert_hw_breakpoint) (gdbarch, bp_tgt)

#define target_remove_hw_breakpoint(gdbarch, bp_tgt) \
     (*current_target.to_remove_hw_breakpoint) (gdbarch, bp_tgt)

/* Return number of debug registers needed for a ranged breakpoint,
   or -1 if ranged breakpoints are not supported.  */

extern int target_ranged_break_num_registers (void);

/* Return non-zero if target knows the data address which triggered this
   target_stopped_by_watchpoint, in such case place it to *ADDR_P.  Only the
   INFERIOR_PTID task is being queried.  */
#define target_stopped_data_address(target, addr_p) \
    (*target.to_stopped_data_address) (target, addr_p)

/* Return non-zero if ADDR is within the range of a watchpoint spanning
   LENGTH bytes beginning at START.  */
#define target_watchpoint_addr_within_range(target, addr, start, length) \
  (*target.to_watchpoint_addr_within_range) (target, addr, start, length)

/* Return non-zero if the target is capable of using hardware to evaluate
   the condition expression.  In this case, if the condition is false when
   the watched memory location changes, execution may continue without the
   debugger being notified.

   Due to limitations in the hardware implementation, it may be capable of
   avoiding triggering the watchpoint in some cases where the condition
   expression is false, but may report some false positives as well.
   For this reason, GDB will still evaluate the condition expression when
   the watchpoint triggers.  */
#define target_can_accel_watchpoint_condition(addr, len, type, cond) \
  (*current_target.to_can_accel_watchpoint_condition) (addr, len, type, cond)

/* Return number of debug registers needed for a masked watchpoint,
   -1 if masked watchpoints are not supported or -2 if the given address
   and mask combination cannot be used.  */

extern int target_masked_watch_num_registers (CORE_ADDR addr, CORE_ADDR mask);

/* Target can execute in reverse?  */
#define target_can_execute_reverse \
     (current_target.to_can_execute_reverse ? \
      current_target.to_can_execute_reverse () : 0)

extern const struct target_desc *target_read_description (struct target_ops *);

#define target_get_ada_task_ptid(lwp, tid) \
     (*current_target.to_get_ada_task_ptid) (lwp,tid)

/* Utility implementation of searching memory.  */
extern int simple_search_memory (struct target_ops* ops,
                                 CORE_ADDR start_addr,
                                 ULONGEST search_space_len,
                                 const gdb_byte *pattern,
                                 ULONGEST pattern_len,
                                 CORE_ADDR *found_addrp);

/* Main entry point for searching memory.  */
extern int target_search_memory (CORE_ADDR start_addr,
                                 ULONGEST search_space_len,
                                 const gdb_byte *pattern,
                                 ULONGEST pattern_len,
                                 CORE_ADDR *found_addrp);

/* Target file operations.  */

/* Open FILENAME on the target, using FLAGS and MODE.  Return a
   target file descriptor, or -1 if an error occurs (and set
   *TARGET_ERRNO).  */
extern int target_fileio_open (const char *filename, int flags, int mode,
			       int *target_errno);

/* Write up to LEN bytes from WRITE_BUF to FD on the target.
   Return the number of bytes written, or -1 if an error occurs
   (and set *TARGET_ERRNO).  */
extern int target_fileio_pwrite (int fd, const gdb_byte *write_buf, int len,
				 ULONGEST offset, int *target_errno);

/* Read up to LEN bytes FD on the target into READ_BUF.
   Return the number of bytes read, or -1 if an error occurs
   (and set *TARGET_ERRNO).  */
extern int target_fileio_pread (int fd, gdb_byte *read_buf, int len,
				ULONGEST offset, int *target_errno);

/* Close FD on the target.  Return 0, or -1 if an error occurs
   (and set *TARGET_ERRNO).  */
extern int target_fileio_close (int fd, int *target_errno);

/* Unlink FILENAME on the target.  Return 0, or -1 if an error
   occurs (and set *TARGET_ERRNO).  */
extern int target_fileio_unlink (const char *filename, int *target_errno);

/* Read value of symbolic link FILENAME on the target.  Return a
   null-terminated string allocated via xmalloc, or NULL if an error
   occurs (and set *TARGET_ERRNO).  */
extern char *target_fileio_readlink (const char *filename, int *target_errno);

/* Read target file FILENAME.  The return value will be -1 if the transfer
   fails or is not supported; 0 if the object is empty; or the length
   of the object otherwise.  If a positive value is returned, a
   sufficiently large buffer will be allocated using xmalloc and
   returned in *BUF_P containing the contents of the object.

   This method should be used for objects sufficiently small to store
   in a single xmalloc'd buffer, when no fixed bound on the object's
   size is known in advance.  */
extern LONGEST target_fileio_read_alloc (const char *filename,
					 gdb_byte **buf_p);

/* Read target file FILENAME.  The result is NUL-terminated and
   returned as a string, allocated using xmalloc.  If an error occurs
   or the transfer is unsupported, NULL is returned.  Empty objects
   are returned as allocated but empty strings.  A warning is issued
   if the result contains any embedded NUL bytes.  */
extern char *target_fileio_read_stralloc (const char *filename);


/* Tracepoint-related operations.  */

#define target_trace_init() \
  (*current_target.to_trace_init) ()

#define target_download_tracepoint(t) \
  (*current_target.to_download_tracepoint) (t)

#define target_can_download_tracepoint() \
  (*current_target.to_can_download_tracepoint) ()

#define target_download_trace_state_variable(tsv) \
  (*current_target.to_download_trace_state_variable) (tsv)

#define target_enable_tracepoint(loc) \
  (*current_target.to_enable_tracepoint) (loc)

#define target_disable_tracepoint(loc) \
  (*current_target.to_disable_tracepoint) (loc)

#define target_trace_start() \
  (*current_target.to_trace_start) ()

#define target_trace_set_readonly_regions() \
  (*current_target.to_trace_set_readonly_regions) ()

#define target_get_trace_status(ts) \
  (*current_target.to_get_trace_status) (ts)

#define target_get_tracepoint_status(tp,utp)		\
  (*current_target.to_get_tracepoint_status) (tp, utp)

#define target_trace_stop() \
  (*current_target.to_trace_stop) ()

#define target_trace_find(type,num,addr1,addr2,tpp) \
  (*current_target.to_trace_find) ((type), (num), (addr1), (addr2), (tpp))

#define target_get_trace_state_variable_value(tsv,val) \
  (*current_target.to_get_trace_state_variable_value) ((tsv), (val))

#define target_save_trace_data(filename) \
  (*current_target.to_save_trace_data) (filename)

#define target_upload_tracepoints(utpp) \
  (*current_target.to_upload_tracepoints) (utpp)

#define target_upload_trace_state_variables(utsvp) \
  (*current_target.to_upload_trace_state_variables) (utsvp)

#define target_get_raw_trace_data(buf,offset,len) \
  (*current_target.to_get_raw_trace_data) ((buf), (offset), (len))

#define target_get_min_fast_tracepoint_insn_len() \
  (*current_target.to_get_min_fast_tracepoint_insn_len) ()

#define target_set_disconnected_tracing(val) \
  (*current_target.to_set_disconnected_tracing) (val)

#define	target_set_circular_trace_buffer(val)	\
  (*current_target.to_set_circular_trace_buffer) (val)

#define	target_set_trace_buffer_size(val)	\
  (*current_target.to_set_trace_buffer_size) (val)

#define	target_set_trace_notes(user,notes,stopnotes)		\
  (*current_target.to_set_trace_notes) ((user), (notes), (stopnotes))

#define target_get_tib_address(ptid, addr) \
  (*current_target.to_get_tib_address) ((ptid), (addr))

#define target_set_permissions() \
  (*current_target.to_set_permissions) ()

#define target_static_tracepoint_marker_at(addr, marker) \
  (*current_target.to_static_tracepoint_marker_at) (addr, marker)

#define target_static_tracepoint_markers_by_strid(marker_id) \
  (*current_target.to_static_tracepoint_markers_by_strid) (marker_id)

#define target_traceframe_info() \
  (*current_target.to_traceframe_info) ()

#define target_use_agent(use) \
  (*current_target.to_use_agent) (use)

#define target_can_use_agent() \
  (*current_target.to_can_use_agent) ()

/* Command logging facility.  */

#define target_log_command(p)						\
  do									\
    if (current_target.to_log_command)					\
      (*current_target.to_log_command) (p);				\
  while (0)


extern int target_core_of_thread (ptid_t ptid);

/* Verify that the memory in the [MEMADDR, MEMADDR+SIZE) range matches
   the contents of [DATA,DATA+SIZE).  Returns 1 if there's a match, 0
   if there's a mismatch, and -1 if an error is encountered while
   reading memory.  Throws an error if the functionality is found not
   to be supported by the current target.  */
int target_verify_memory (const gdb_byte *data,
			  CORE_ADDR memaddr, ULONGEST size);

/* Routines for maintenance of the target structures...

   add_target:   Add a target to the list of all possible targets.

   push_target:  Make this target the top of the stack of currently used
   targets, within its particular stratum of the stack.  Result
   is 0 if now atop the stack, nonzero if not on top (maybe
   should warn user).

   unpush_target: Remove this from the stack of currently used targets,
   no matter where it is on the list.  Returns 0 if no
   change, 1 if removed from stack.

   pop_target:   Remove the top thing on the stack of current targets.  */

extern struct cmd_list_element *add_target (struct target_ops *);

/* Adds a command ALIAS for target T and marks it deprecated.  This is useful
   for maintaining backwards compatibility when renaming targets.  */

extern void add_deprecated_target_alias (struct target_ops *t, char *alias);

extern void push_target (struct target_ops *);

extern int unpush_target (struct target_ops *);

extern void target_pre_inferior (int);

extern void target_preopen (int);

extern void pop_target (void);

/* Does whatever cleanup is required to get rid of all pushed targets.
   QUITTING is propagated to target_close; it indicates that GDB is
   exiting and should not get hung on an error (otherwise it is
   important to perform clean termination, even if it takes a
   while).  */
extern void pop_all_targets (int quitting);

/* Like pop_all_targets, but pops only targets whose stratum is
   strictly above ABOVE_STRATUM.  */
extern void pop_all_targets_above (enum strata above_stratum, int quitting);

extern int target_is_pushed (struct target_ops *t);

extern CORE_ADDR target_translate_tls_address (struct objfile *objfile,
					       CORE_ADDR offset);

/* Struct target_section maps address ranges to file sections.  It is
   mostly used with BFD files, but can be used without (e.g. for handling
   raw disks, or files not in formats handled by BFD).  */

struct target_section
  {
    CORE_ADDR addr;		/* Lowest address in section */
    CORE_ADDR endaddr;		/* 1+highest address in section */

    struct bfd_section *the_bfd_section;

    /* A given BFD may appear multiple times in the target section
       list, so each BFD is associated with a given key.  The key is
       just some convenient pointer that can be used to differentiate
       the BFDs.  These are managed only by convention.  */
    void *key;

    bfd *bfd;			/* BFD file pointer */
  };

/* Holds an array of target sections.  Defined by [SECTIONS..SECTIONS_END[.  */

struct target_section_table
{
  struct target_section *sections;
  struct target_section *sections_end;
};

/* Return the "section" containing the specified address.  */
struct target_section *target_section_by_addr (struct target_ops *target,
					       CORE_ADDR addr);

/* Return the target section table this target (or the targets
   beneath) currently manipulate.  */

extern struct target_section_table *target_get_section_table
  (struct target_ops *target);

/* From mem-break.c */

extern int memory_remove_breakpoint (struct gdbarch *,
				     struct bp_target_info *);

extern int memory_insert_breakpoint (struct gdbarch *,
				     struct bp_target_info *);

extern int default_memory_remove_breakpoint (struct gdbarch *,
					     struct bp_target_info *);

extern int default_memory_insert_breakpoint (struct gdbarch *,
					     struct bp_target_info *);


/* From target.c */

extern void initialize_targets (void);

extern void noprocess (void) ATTRIBUTE_NORETURN;

extern void target_require_runnable (void);

extern void find_default_attach (struct target_ops *, char *, int);

extern void find_default_create_inferior (struct target_ops *,
					  char *, char *, char **, int);

extern struct target_ops *find_run_target (void);

extern struct target_ops *find_target_beneath (struct target_ops *);

/* Read OS data object of type TYPE from the target, and return it in
   XML format.  The result is NUL-terminated and returned as a string,
   allocated using xmalloc.  If an error occurs or the transfer is
   unsupported, NULL is returned.  Empty objects are returned as
   allocated but empty strings.  */

extern char *target_get_osdata (const char *type);


/* Stuff that should be shared among the various remote targets.  */

/* Debugging level.  0 is off, and non-zero values mean to print some debug
   information (higher values, more information).  */
extern int remote_debug;

/* Speed in bits per second, or -1 which means don't mess with the speed.  */
extern int baud_rate;
/* Timeout limit for response from target.  */
extern int remote_timeout;



/* Set the show memory breakpoints mode to show, and installs a cleanup
   to restore it back to the current value.  */
extern struct cleanup *make_show_memory_breakpoints_cleanup (int show);

extern int may_write_registers;
extern int may_write_memory;
extern int may_insert_breakpoints;
extern int may_insert_tracepoints;
extern int may_insert_fast_tracepoints;
extern int may_stop;

extern void update_target_permissions (void);


/* Imported from machine dependent code.  */

/* Blank target vector entries are initialized to target_ignore.  */
void target_ignore (void);

/* See to_supports_btrace in struct target_ops.  */
extern int target_supports_btrace (void);

/* See to_enable_btrace in struct target_ops.  */
extern struct btrace_target_info *target_enable_btrace (ptid_t ptid);

/* See to_disable_btrace in struct target_ops.  */
extern void target_disable_btrace (struct btrace_target_info *btinfo);

/* See to_teardown_btrace in struct target_ops.  */
extern void target_teardown_btrace (struct btrace_target_info *btinfo);

/* See to_read_btrace in struct target_ops.  */
extern VEC (btrace_block_s) *target_read_btrace (struct btrace_target_info *,
						 enum btrace_read_type);

/* See to_stop_recording in struct target_ops.  */
extern void target_stop_recording (void);

/* See to_info_record in struct target_ops.  */
extern void target_info_record (void);

/* See to_save_record in struct target_ops.  */
extern void target_save_record (char *filename);

/* Query if the target supports deleting the execution log.  */
extern int target_supports_delete_record (void);

/* See to_delete_record in struct target_ops.  */
extern void target_delete_record (void);

/* See to_record_is_replaying in struct target_ops.  */
extern int target_record_is_replaying (void);

/* See to_goto_record_begin in struct target_ops.  */
extern void target_goto_record_begin (void);

/* See to_goto_record_end in struct target_ops.  */
extern void target_goto_record_end (void);

/* See to_goto_record in struct target_ops.  */
extern void target_goto_record (ULONGEST insn);

/* See to_insn_history.  */
extern void target_insn_history (int size, int flags);

/* See to_insn_history_from.  */
extern void target_insn_history_from (ULONGEST from, int size, int flags);

/* See to_insn_history_range.  */
extern void target_insn_history_range (ULONGEST begin, ULONGEST end, int flags);

/* See to_call_history.  */
extern void target_call_history (int size, int flags);

/* See to_call_history_from.  */
extern void target_call_history_from (ULONGEST begin, int size, int flags);

/* See to_call_history_range.  */
extern void target_call_history_range (ULONGEST begin, ULONGEST end, int flags);

#endif /* !defined (TARGET_H) */
