/* UI_FILE - a generic STDIO like output stream.
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

#ifndef UI_FILE_H
#define UI_FILE_H

struct obstack;
struct ui_file;

/* Create a generic ui_file object with null methods.  */

extern struct ui_file *ui_file_new (void);

/* Override methods used by specific implementations of a UI_FILE
   object.  */

typedef void (ui_file_flush_ftype) (struct ui_file *stream);
extern void set_ui_file_flush (struct ui_file *stream,
			       ui_file_flush_ftype *flush);

/* NOTE: Both fputs and write methods are available.  Default
   implementations that mapping one onto the other are included.  */
typedef void (ui_file_write_ftype) (struct ui_file *stream,
				    const char *buf, long length_buf);
extern void set_ui_file_write (struct ui_file *stream,
			       ui_file_write_ftype *fputs);

typedef void (ui_file_fputs_ftype) (const char *, struct ui_file *stream);
extern void set_ui_file_fputs (struct ui_file *stream,
			       ui_file_fputs_ftype *fputs);

/* This version of "write" is safe for use in signal handlers.
   It's not guaranteed that all existing output will have been
   flushed first.
   Implementations are also free to ignore some or all of the request.
   fputs_async is not provided as the async versions are rarely used,
   no point in having both for a rarely used interface.  */
typedef void (ui_file_write_async_safe_ftype)
  (struct ui_file *stream, const char *buf, long length_buf);
extern void set_ui_file_write_async_safe
  (struct ui_file *stream, ui_file_write_async_safe_ftype *write_async_safe);

typedef long (ui_file_read_ftype) (struct ui_file *stream,
				   char *buf, long length_buf);
extern void set_ui_file_read (struct ui_file *stream,
			      ui_file_read_ftype *fread);

typedef int (ui_file_isatty_ftype) (struct ui_file *stream);
extern void set_ui_file_isatty (struct ui_file *stream,
				ui_file_isatty_ftype *isatty);

typedef void (ui_file_rewind_ftype) (struct ui_file *stream);
extern void set_ui_file_rewind (struct ui_file *stream,
				ui_file_rewind_ftype *rewind);

typedef void (ui_file_put_method_ftype) (void *object, const char *buffer,
					 long length_buffer);
typedef void (ui_file_put_ftype) (struct ui_file *stream,
				  ui_file_put_method_ftype *method,
				  void *context);
extern void set_ui_file_put (struct ui_file *stream, ui_file_put_ftype *put);

typedef void (ui_file_delete_ftype) (struct ui_file * stream);
extern void set_ui_file_data (struct ui_file *stream, void *data,
			      ui_file_delete_ftype *delete);

typedef int (ui_file_fseek_ftype) (struct ui_file *stream, long offset,
				   int whence);
extern void set_ui_file_fseek (struct ui_file *stream,
			       ui_file_fseek_ftype *fseek_ptr);

extern void *ui_file_data (struct ui_file *file);


extern void gdb_flush (struct ui_file *);

extern void ui_file_delete (struct ui_file *stream);

extern void ui_file_rewind (struct ui_file *stream);

extern int ui_file_isatty (struct ui_file *);

extern void ui_file_write (struct ui_file *file, const char *buf,
			   long length_buf);

extern void ui_file_write_async_safe (struct ui_file *file, const char *buf,
				      long length_buf);

/* NOTE: copies left to right.  */
extern void ui_file_put (struct ui_file *src,
			 ui_file_put_method_ftype *write, void *dest);

/* Returns a freshly allocated buffer containing the entire contents
   of FILE (as determined by ui_file_put()) with a NUL character
   appended.  LENGTH, if not NULL, is set to the size of the buffer
   minus that appended NUL.  */
extern char *ui_file_xstrdup (struct ui_file *file, long *length);

/* Similar to ui_file_xstrdup, but return a new string allocated on
   OBSTACK.  */
extern char *ui_file_obsavestring (struct ui_file *file,
				   struct obstack *obstack, long *length);

extern long ui_file_read (struct ui_file *file, char *buf, long length_buf);

extern int ui_file_fseek (struct ui_file *file, long offset, int whence);

/* Create/open a memory based file.  Can be used as a scratch buffer
   for collecting output.  */
extern struct ui_file *mem_fileopen (void);



/* Open/create a STDIO based UI_FILE using the already open FILE.  */
extern struct ui_file *stdio_fileopen (FILE *file);

/* Open NAME returning an STDIO based UI_FILE.  */
extern struct ui_file *gdb_fopen (char *name, char *mode);

/* Create a file which writes to both ONE and TWO.  CLOSE_ONE
   and CLOSE_TWO indicate whether the original files should be
   closed when the new file is closed.  */
extern struct ui_file *tee_file_new (struct ui_file *one,
				     int close_one,
				     struct ui_file *two,
				     int close_two);
#endif
