/* Top level stuff for GDB, the GNU debugger.

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

#ifndef TOP_H
#define TOP_H

/* From top.c.  */
extern char *saved_command_line;
extern int saved_command_line_size;
extern FILE *instream;
extern int in_user_command;
extern int confirm;
extern char gdb_dirbuf[1024];
extern int inhibit_gdbinit;
extern char gdbinit[];

extern void print_gdb_version (struct ui_file *);

extern void read_command_file (FILE *);
extern void init_history (void);
extern void command_loop (void);
extern int quit_confirm (void);
extern void quit_force (char *, int);
extern void quit_command (char *, int);
extern void quit_cover (void);
extern void execute_command (char *, int);

extern void check_frame_language_change (void);

/* Prepare for execution of a command.
   Call this before every command, CLI or MI.
   Returns a cleanup to be run after the command is completed.  */
extern struct cleanup *prepare_execute_command (void);

/* This function returns a pointer to the string that is used
   by gdb for its command prompt.  */
extern char *get_prompt (void);

/* This function returns a pointer to the string that is used
   by gdb for its command prompt.  */
extern void set_prompt (const char *s);

/* From random places.  */
extern int readnow_symbol_files;

/* Perform _initialize initialization.  */
extern void gdb_init (char *);

/* For use by event-top.c.  */
/* Variables from top.c.  */
extern int source_line_number;
extern const char *source_file_name;
extern int history_expansion_p;
extern int server_command;
extern char *lim_at_start;

extern void show_commands (char *args, int from_tty);

extern void set_history (char *, int);

extern void show_history (char *, int);

extern void set_verbose (char *, int, struct cmd_list_element *);

extern void do_restore_instream_cleanup (void *stream);

#endif
