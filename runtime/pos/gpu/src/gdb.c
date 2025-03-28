/* Top level stuff for GDB, the GNU debugger.

   Copyright (C) 1986-2019 Free Software Foundation, Inc.

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
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2020 NVIDIA Corporation
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

#include "defs.h"
#include "top.h"
#include "target.h"
#include "inferior.h"
#include "symfile.h"
#include "gdbcore.h"
#include "getopt.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <ctype.h>
#include "event-loop.h"
#include "ui-out.h"

#include "interps.h"
#include "main.h"
#include "source.h"
#include "cli/cli-cmds.h"
#include "objfiles.h"
#include "auto-load.h"
#include "maint.h"
#include "cuda-gdb.h"

#include "filenames.h"
#include "common/filestuff.h"
#include <signal.h>
#include "event-top.h"
#include "infrun.h"
#include "common/signals-state-save-restore.h"
#include <vector>
#include "common/pathstuff.h"
#include "cli/cli-style.h"

/* The selected interpreter.  This will be used as a set command
   variable, so it should always be malloc'ed - since
   do_setshow_command will free it.  */
char *interpreter_p;

/* Whether dbx commands will be handled.  */
int dbx_commands = 0;

/* System root path, used to find libraries etc.  */
char *gdb_sysroot = 0;

/* GDB datadir, used to store data files.  */
char *gdb_datadir = 0;

/* Non-zero if GDB_DATADIR was provided on the command line.
   This doesn't track whether data-directory is set later from the
   command line, but we don't reread system.gdbinit when that happens.  */
static int gdb_datadir_provided = 0;

/* If gdb was configured with --with-python=/path,
   the possibly relocated path to python's lib directory.  */
char *python_libdir = 0;

/* Target IO streams.  */
struct ui_file *gdb_stdtargin;
struct ui_file *gdb_stdtarg;
struct ui_file *gdb_stdtargerr;

/* True if --batch or --batch-silent was seen.  */
int batch_flag = 1;

/* Support for the --batch-silent option.  */
int batch_silent = 0;

/* Support for --return-child-result option.
   Set the default to -1 to return error in the case
   that the program does not run or does not complete.  */
int return_child_result = 0;
int return_child_result_value = -1;


/* GDB as it has been invoked from the command line (i.e. argv[0]).  */
char *gdb_program_name;

/* Return read only pointer to GDB_PROGRAM_NAME.  */
const char *
get_gdb_program_name (void)
{
  return gdb_program_name;
}


/* Set the data-directory parameter to NEW_DATADIR.
   If NEW_DATADIR is not a directory then a warning is printed.
   We don't signal an error for backward compatibility.  */

void
set_gdb_data_directory (const char *new_datadir)
{
  struct stat st;

  if (stat (new_datadir, &st) < 0)
    {
      int save_errno = errno;

      fprintf_unfiltered (gdb_stderr, "Warning: ");
      print_sys_errmsg (new_datadir, save_errno);
    }
  else if (!S_ISDIR (st.st_mode))
    warning (_("%s is not a directory."), new_datadir);

  xfree (gdb_datadir);
  gdb_datadir = gdb_realpath (new_datadir).release ();

  /* gdb_realpath won't return an absolute path if the path doesn't exist,
     but we still want to record an absolute path here.  If the user entered
     "../foo" and "../foo" doesn't exist then we'll record $(pwd)/../foo which
     isn't canonical, but that's ok.  */
  if (!IS_ABSOLUTE_PATH (gdb_datadir))
    {
      gdb::unique_xmalloc_ptr<char> abs_datadir = gdb_abspath (gdb_datadir);

      xfree (gdb_datadir);
      gdb_datadir = abs_datadir.release ();
    }
}

/* Relocate a file or directory.  PROGNAME is the name by which gdb
   was invoked (i.e., argv[0]).  INITIAL is the default value for the
   file or directory.  FLAG is true if the value is relocatable, false
   otherwise.  Returns a newly allocated string; this may return NULL
   under the same conditions as make_relative_prefix.  */

static char *
relocate_path (const char *progname, const char *initial, int flag)
{
  if (flag)
    return make_relative_prefix (progname, BINDIR, initial);
  return xstrdup (initial);
}

/* Like relocate_path, but specifically checks for a directory.
   INITIAL is relocated according to the rules of relocate_path.  If
   the result is a directory, it is used; otherwise, INITIAL is used.
   The chosen directory is then canonicalized using lrealpath.  This
   function always returns a newly-allocated string.  */

char *
relocate_gdb_directory (const char *initial, int flag)
{
  char *dir;

  dir = relocate_path (gdb_program_name, initial, flag);
  if (dir)
    {
      struct stat s;

      if (*dir == '\0' || stat (dir, &s) != 0 || !S_ISDIR (s.st_mode))
	{
	  xfree (dir);
	  dir = NULL;
	}
    }
  if (!dir)
    dir = xstrdup (initial);

  /* Canonicalize the directory.  */
  if (*dir)
    {
      char *canon_sysroot = lrealpath (dir);

      if (canon_sysroot)
	{
	  xfree (dir);
	  dir = canon_sysroot;
	}
    }

  return dir;
}

/* Compute the locations of init files that GDB should source and
   return them in SYSTEM_GDBINIT, HOME_GDBINIT, LOCAL_GDBINIT.  If
   there is no system gdbinit (resp. home gdbinit and local gdbinit)
   to be loaded, then SYSTEM_GDBINIT (resp. HOME_GDBINIT and
   LOCAL_GDBINIT) is set to NULL.  */
static void
get_init_files (const char **system_gdbinit,
		const char **home_gdbinit,
		const char **local_gdbinit)
{
  static const char *sysgdbinit = NULL;
  static char *homeinit = NULL;
  static const char *localinit = NULL;
  static int initialized = 0;

  if (!initialized)
    {
      struct stat homebuf, cwdbuf, s;
      const char *homedir;

      if (SYSTEM_GDBINIT[0])
	{
	  int datadir_len = strlen (GDB_DATADIR);
	  int sys_gdbinit_len = strlen (SYSTEM_GDBINIT);
	  char *relocated_sysgdbinit;

	  /* If SYSTEM_GDBINIT lives in data-directory, and data-directory
	     has been provided, search for SYSTEM_GDBINIT there.  */
	  if (gdb_datadir_provided
	      && datadir_len < sys_gdbinit_len
	      && filename_ncmp (SYSTEM_GDBINIT, GDB_DATADIR, datadir_len) == 0
	      && IS_DIR_SEPARATOR (SYSTEM_GDBINIT[datadir_len]))
	    {
	      /* Append the part of SYSTEM_GDBINIT that follows GDB_DATADIR
		 to gdb_datadir.  */
	      char *tmp_sys_gdbinit = xstrdup (&SYSTEM_GDBINIT[datadir_len]);
	      char *p;

	      for (p = tmp_sys_gdbinit; IS_DIR_SEPARATOR (*p); ++p)
		continue;
	      relocated_sysgdbinit = concat (gdb_datadir, SLASH_STRING, p,
					     (char *) NULL);
	      xfree (tmp_sys_gdbinit);
	    }
	  else
	    {
	      relocated_sysgdbinit = relocate_path (gdb_program_name,
						    SYSTEM_GDBINIT,
						    SYSTEM_GDBINIT_RELOCATABLE);
	    }
	  if (relocated_sysgdbinit && stat (relocated_sysgdbinit, &s) == 0)
	    sysgdbinit = relocated_sysgdbinit;
	  else
	    xfree (relocated_sysgdbinit);
	}

      homedir = getenv ("HOME");

      /* If the .gdbinit file in the current directory is the same as
	 the $HOME/.gdbinit file, it should not be sourced.  homebuf
	 and cwdbuf are used in that purpose.  Make sure that the stats
	 are zero in case one of them fails (this guarantees that they
	 won't match if either exists).  */

      memset (&homebuf, 0, sizeof (struct stat));
      memset (&cwdbuf, 0, sizeof (struct stat));

      if (homedir)
	{
	  homeinit = xstrprintf ("%s/%s", homedir, gdbinit);
	  if (stat (homeinit, &homebuf) != 0)
	    {
	      xfree (homeinit);
	      homeinit = NULL;
	    }
	}

      if (stat (gdbinit, &cwdbuf) == 0)
	{
	  if (!homeinit
	      || memcmp ((char *) &homebuf, (char *) &cwdbuf,
			 sizeof (struct stat)))
	    localinit = gdbinit;
	}
      
      initialized = 1;
    }

  *system_gdbinit = sysgdbinit;
  *home_gdbinit = homeinit;
  *local_gdbinit = localinit;
}

/* Try to set up an alternate signal stack for SIGSEGV handlers.
   This allows us to handle SIGSEGV signals generated when the
   normal process stack is exhausted.  If this stack is not set
   up (sigaltstack is unavailable or fails) and a SIGSEGV is
   generated when the normal stack is exhausted then the program
   will behave as though no SIGSEGV handler was installed.  */

static void
setup_alternate_signal_stack (void)
{
  stack_t ss;

  /* FreeBSD versions older than 11.0 use char * for ss_sp instead of
     void *.  This cast works with both types.  */
  ss.ss_sp = (char *) xmalloc (SIGSTKSZ);
  ss.ss_size = SIGSTKSZ;
  ss.ss_flags = 0;

  sigaltstack(&ss, NULL);
}

/* Call command_loop.  */

/* Prevent inlining this function for the benefit of GDB's selftests
   in the testsuite.  Those tests want to run GDB under GDB and stop
   here.  */
static void captured_command_loop () __attribute__((noinline));

static void
captured_command_loop ()
{
  struct ui *ui = current_ui;

  /* Top-level execution commands can be run in the background from
     here on.  */
  current_ui->async = 1;

  /* Give the interpreter a chance to print a prompt, if necessary  */
  if (ui->prompt_state != PROMPT_BLOCKED)
    interp_pre_command_loop (top_level_interpreter ());

  /* Now it's time to start the event loop.  */
  start_event_loop ();

  /* FIXME: cagney/1999-11-05: A correct command_loop() implementaton
     would clean things up (restoring the cleanup chain) to the state
     they were just prior to the call.  Technically, this means that
     the do_cleanups() below is redundant.  Unfortunately, many FUNCs
     are not that well behaved.  do_cleanups should either be replaced
     with a do_cleanups call (to cover the problem) or an assertion
     check to detect bad FUNCs code.  */
  do_cleanups (all_cleanups ());
  /* If the command_loop returned, normally (rather than threw an
     error) we try to quit.  If the quit is aborted, our caller
     catches the signal and restarts the command loop.  */
  quit_command (NULL, ui->instream == ui->stdin_stream);
}

/* Handle command errors thrown from within catch_command_errors.  */

static int
handle_command_errors (struct gdb_exception e)
{
  if (e.reason < 0)
    {
      exception_print (gdb_stderr, e);

      /* If any exception escaped to here, we better enable stdin.
	 Otherwise, any command that calls async_disable_stdin, and
	 then throws, will leave stdin inoperable.  */
      async_enable_stdin ();
      return 0;
    }
  return 1;
}

/* Type of the command callback passed to the const
   catch_command_errors.  */

typedef void (catch_command_errors_const_ftype) (const char *, int);

/* Wrap calls to commands run before the event loop is started.  */

static int
catch_command_errors (catch_command_errors_const_ftype command,
		      const char *arg, int from_tty)
{
  TRY
    {
      int was_sync = current_ui->prompt_state == PROMPT_BLOCKED;

      command (arg, from_tty);

      maybe_wait_sync_command_done (was_sync);
    }
  CATCH (e, RETURN_MASK_ALL)
    {
      return handle_command_errors (e);
    }
  END_CATCH

  return 1;
}

/* Adapter for symbol_file_add_main that translates 'from_tty' to a
   symfile_add_flags.  */

static void
symbol_file_add_main_adapter (const char *arg, int from_tty)
{
  symfile_add_flags add_flags = 0;

  if (from_tty)
    add_flags |= SYMFILE_VERBOSE;

  symbol_file_add_main (arg, add_flags);
}

/* Perform validation of the '--readnow' and '--readnever' flags.  */

static void
validate_readnow_readnever ()
{
  if (readnever_symbol_files && readnow_symbol_files)
    {
      error (_("%s: '--readnow' and '--readnever' cannot be "
	       "specified simultaneously"),
	     gdb_program_name);
    }
}

/* Type of this option.  */
enum cmdarg_kind
{
  /* Option type -x.  */
  CMDARG_FILE,

  /* Option type -ex.  */
  CMDARG_COMMAND,

  /* Option type -ix.  */
  CMDARG_INIT_FILE,
    
  /* Option type -iex.  */
  CMDARG_INIT_COMMAND
};

/* Arguments of --command option and its counterpart.  */
struct cmdarg
{
  cmdarg (cmdarg_kind type_, char *string_)
    : type (type_), string (string_)
  {}

  /* Type of this option.  */
  enum cmdarg_kind type;

  /* Value of this option - filename or the GDB command itself.  String memory
     is not owned by this structure despite it is 'const'.  */
  char *string;
};

int gdb_init (int argc, char **argv, char *execarg, char* pidarg)
{
  static int quiet = 0;
  static int set_args = 0;
  static int inhibit_home_gdbinit = 0;

  /* Pointers to various arguments from command line.  */
  char *symarg = execarg;
  char *cdarg = NULL;
  char *ttyarg = NULL;

  /* These are static so that we can take their address in an
     initializer.  */
  static int print_help;
  static int print_version;
  static int print_configuration;

  /* Pointers to all arguments of --command option.  */
  std::vector<struct cmdarg> cmdarg_vec;

  /* All arguments of --directory option.  */
  std::vector<char *> dirarg;

  /* gdb init files.  */
  const char *system_gdbinit;
  const char *home_gdbinit;
  const char *local_gdbinit;

  int i;
  int save_auto_load;
  int ret = 1;

#ifdef HAVE_USEFUL_SBRK
  /* Set this before constructing scoped_command_stats.  */
# ifdef __clang__
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wdeprecated-declarations"
  lim_at_start = (char *) sbrk (0);
#   pragma clang diagnostic pop
# else
  lim_at_start = (char *) sbrk (0);
# endif /* __clang__ */
#endif

  scoped_command_stats stat_reporter (false);

  setlocale (LC_MESSAGES, "");
  setlocale (LC_CTYPE, "");
  bindtextdomain (PACKAGE, "/usr/local/share/locale");
  textdomain (PACKAGE);

  notice_open_fds ();

  saved_command_line = (char *) xstrdup ("");

  /* Note: `error' cannot be called before this point, because the
     caller will crash when trying to print the exception.  */
  main_ui = new ui (stdin, stdout, stderr);
  current_ui = main_ui;

  gdb_stdtargerr = gdb_stderr;	/* for moment */
  gdb_stdtargin = gdb_stdin;	/* for moment */

  if (bfd_init () != BFD_INIT_MAGIC)
    error (_("fatal error: libbfd ABI mismatch"));

  gdb_program_name = xstrdup (argv[0]);

  /* Prefix warning messages with the command name.  */
  gdb::unique_xmalloc_ptr<char> tmp_warn_preprint
    (xstrprintf ("%s: warning: ", gdb_program_name));
  warning_pre_print = tmp_warn_preprint.get ();

  current_directory = getcwd (NULL, 0);
  if (current_directory == NULL)
    perror_warning_with_name (_("error finding working directory"));

  /* Set the sysroot path.  */
  gdb_sysroot = relocate_gdb_directory (TARGET_SYSTEM_ROOT,
					TARGET_SYSTEM_ROOT_RELOCATABLE);

  if (gdb_sysroot == NULL || *gdb_sysroot == '\0')
    {
      xfree (gdb_sysroot);
      gdb_sysroot = xstrdup (TARGET_SYSROOT_PREFIX);
    }

  debug_file_directory = relocate_gdb_directory (DEBUGDIR,
						 DEBUGDIR_RELOCATABLE);

  gdb_datadir = relocate_gdb_directory (GDB_DATADIR,
					GDB_DATADIR_RELOCATABLE);

  /* There will always be an interpreter.  Either the one passed into
     this captured main, or one specified by the user at start up, or
     the console.  Initialize the interpreter to the one requested by 
     the application.  */
  interpreter_p = xstrdup (INTERP_CONSOLE);

  quiet = 1;

  /* Disable all output styling when running in batch mode.  */
  cli_styling = 0;

  save_original_signals_state (quiet);

  /* Try to set up an alternate signal stack for SIGSEGV handlers.  */
  setup_alternate_signal_stack ();

  /* Initialize all files.  */
  gdb_init (gdb_program_name);


  /* Lookup gdbinit files.  Note that the gdbinit file name may be
     overriden during file initialization, so get_init_files should be
     called after gdb_init.  */
  get_init_files (&system_gdbinit, &home_gdbinit, &local_gdbinit);

  /* Do these (and anything which might call wrap_here or *_filtered)
     after initialize_all_files() but before the interpreter has been
     installed.  Otherwize the help/version messages will be eaten by
     the interpreter's output handler.  */

  /* Install the default UI.  All the interpreters should have had a
     look at things by now.  Initialize the default interpreter.  */
  set_top_level_interpreter (interpreter_p);

  /* Set off error and warning messages with a blank line.  */
  tmp_warn_preprint.reset ();
  warning_pre_print = _("\nwarning: ");

  /* Read and execute the system-wide gdbinit file, if it exists.
     This is done *before* all the command line arguments are
     processed; it sets global parameters, which are independent of
     what file you are debugging or what directory you are in.  */
  if (system_gdbinit && !inhibit_gdbinit)
    ret = catch_command_errors (source_script, system_gdbinit, 0);

  /* Read and execute $HOME/.gdbinit file, if it exists.  This is done
     *before* all the command line arguments are processed; it sets
     global parameters, which are independent of what file you are
     debugging or what directory you are in.  */

  if (home_gdbinit && !inhibit_gdbinit && !inhibit_home_gdbinit)
    ret = catch_command_errors (source_script, home_gdbinit, 0);

  /* Skip auto-loading section-specified scripts until we've sourced
     local_gdbinit (which is often used to augment the source search
     path).  */
  save_auto_load = global_auto_load;
  global_auto_load = 0;

  if (execarg != NULL
      && symarg != NULL
      && strcmp (execarg, symarg) == 0)
    {
      /* The exec file and the symbol-file are the same.  If we can't
         open it, better only print one error message.
         catch_command_errors returns non-zero on success!  */
      ret = catch_command_errors (exec_file_attach, execarg,
				  !batch_flag);
      if (ret != 0)
	ret = catch_command_errors (symbol_file_add_main_adapter,
				    symarg, !batch_flag);
    }
  else
    {
      if (execarg != NULL)
	ret = catch_command_errors (exec_file_attach, execarg,
				    !batch_flag);
      if (symarg != NULL)
	ret = catch_command_errors (symbol_file_add_main_adapter,
				    symarg, !batch_flag);
    }

  if (pidarg != NULL)
    {
      ret = catch_command_errors (attach_command, pidarg, !batch_flag);
    }

  /* Error messages should no longer be distinguished with extra output.  */
  warning_pre_print = _("warning: ");

  /* Now that all .gdbinit's have been read and all -d options have been
     processed, we can read any scripts mentioned in SYMARG.
     We wait until now because it is common to add to the source search
     path in local_gdbinit.  */
  global_auto_load = save_auto_load;
  for (objfile *objfile : current_program_space->objfiles ())
    load_auto_scripts_for_objfile (objfile);

  //ret = catch_command_errors (execute_command, "run",
  //                !batch_flag);

  /* Read in the old history after all the command files have been
     read.  */
  init_history ();
  return 0;
}
