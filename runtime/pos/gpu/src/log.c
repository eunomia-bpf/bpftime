// The MIT License (MIT)
// 
// Copyright (c) 2012 Niklas E.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "log.h"

#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdarg.h>
#include <string.h>

struct log_data* get_log_data() {
	static struct log_data log_data;
	return &log_data;
}

int str_find_last_of(const char* to_search, char to_find) {
	int len = strlen(to_search);
	for(int i = len-1; i >= 0; --i) {
		if(to_search[i] == to_find)
			return i;
	}
	return 0;
}

void str_strip(char* to_strip, int offset) {
	int len = strlen(to_strip);
	for(int i = offset; i < len; ++i) {
		to_strip[i-offset] = to_strip[i];
	}
	to_strip[len-offset] = '\0';
}

void init_log(char log_level, const char* proj_root)
{
	get_log_data()->curr_level=log_level;
	get_log_data()->project_offset = str_find_last_of(proj_root, '/');
}

void now_time(char* buf)
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	char buffer[100];
	strftime(buffer, sizeof(buffer), "%x %X", localtime(&tv.tv_sec));
	sprintf(buf, "%s.%06ld", buffer, (long)tv.tv_usec);
}

const char* to_string(log_level level)
{
	static const char* const buffer[] = {"ERROR", "WARNING", "INFO", "DEBUG"};
	if(level > LOG_DEBUG){
		return buffer[LOG_DEBUG];
	}
	else return buffer[(int)level];
}

void loggf(log_level level, const char* formatstr, ... )
{
	va_list vararg;
	va_start(vararg, formatstr);
	
	char time[100];
	now_time(time);
	printf("%s (%s):\t", time, to_string(level));
	vprintf(formatstr, vararg);
	printf("\n");
}

void loggfe(log_level level, int line, const char* file, const char* formatstr, ... )
{
	va_list vararg;
	va_start(vararg, formatstr);
	
	char time[64];
	now_time(time);
	printf("%s %7s: ", time, to_string(level));
	vprintf(formatstr, vararg);
	char stripped[64];
	strcpy(stripped, file);
	str_strip(stripped, get_log_data()->project_offset);
	printf("\tin %s(%d)\n", stripped, line);
}
