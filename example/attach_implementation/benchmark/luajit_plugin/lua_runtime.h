#ifndef LUA_RUNTIME_H
#define LUA_RUNTIME_H

#include <stdbool.h>
#include <stdint.h>

/**
 * Initialize the Lua runtime and load the URL filter module
 * 
 * @param prefix The URL prefix to filter (can be NULL for default)
 * @return 0 on success, -1 on failure
 */
int initialize(const char *prefix);

/**
 * Filter a URL based on the loaded Lua module logic
 * 
 * @param url The URL to filter
 * @return true if the URL is allowed, false if it should be blocked
 */
bool url_filter(const char *url);

/**
 * Get the counters for accepted and rejected URLs
 * 
 * @param accepted Pointer to store the number of accepted URLs
 * @param rejected Pointer to store the number of rejected URLs
 */
void get_counters(uint64_t *accepted, uint64_t *rejected);

/**
 * Set data in the Lua module's buffer
 * 
 * @param data The data to copy into the buffer
 * @param size Size of the data
 * @return 0 on success, -1 on failure
 */
int set_buffer(const char *data, size_t size);

/**
 * Get data from the Lua module's buffer
 * 
 * @param out_data Pointer to store the result data
 * @param out_size Pointer to store the size of the data
 * @return 0 on success, -1 on failure
 */
int get_buffer(const char **out_data, size_t *out_size);

#endif /* LUA_RUNTIME_H */ 