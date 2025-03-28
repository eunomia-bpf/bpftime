/* Header file for GDB CLI set and show commands implementation.
   Copyright (C) 2000-2013 Free Software Foundation, Inc.

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

#if !defined (CLI_SETSHOW_H)
#define CLI_SETSHOW_H 1

struct cmd_list_element;

/* Exported to cli/cli-cmds.c and gdb/top.c */

extern void do_set_command (char *arg, int from_tty,
			    struct cmd_list_element *c);
extern void do_show_command (char *arg, int from_tty,
			     struct cmd_list_element *c);

/* Exported to cli/cli-cmds.c and gdb/top.c, language.c and valprint.c */

extern void cmd_show_list (struct cmd_list_element *list, int from_tty,
			   char *prefix);

#endif /* !defined (CLI_SETSHOW_H) */
