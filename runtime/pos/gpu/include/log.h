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
// 
// Logging class by Niklas E.
// usage: call init_log() somewhere early. Pass the logging-level you wish to log and the __FILE__-macro.
// 		To log: call LOG() or LOGE() macros and pass a logging-level and printf-style data (e.g. a format-string as second argument, followed by data),
// tested and developed with gcc 4.6.3

#ifndef __LOG_H__
#define __LOG_H__

#define IFLOG(level) \
if (level > get_log_data()->curr_level) ; \
else 

#define LOG(level, ...) \
if (level > get_log_data()->curr_level) ; \
else loggf(level, __VA_ARGS__)

#define LOGE(level, ...) \
if (level > get_log_data()->curr_level) ; \
else loggfe(level, __LINE__, __FILE__, __VA_ARGS__)

#define LOG_ERROR 0
#define LOG_WARNING 1
#define LOG_INFO 2
#define LOG_DEBUG 3
#define LOG_DBG(i) LOG_DEBUG + i

typedef char log_level;

struct log_data{
	log_level curr_level;
	int project_offset;
};

//log_level: the level to log (see macros above)
//proj_root: a file in the project root. please pass the __FILE__ macro here.
void init_log(char log_level, const char* proj_root);

struct log_data* get_log_data();

void loggf(log_level level, const char* formatstr, ... );

//filenames should not exceed 64 characters
//time should not exceed 64 characters
void loggfe(log_level level, int line, const char* file, const char* formatstr, ... );

#endif //__LOG_H__
