#ifndef WASM_RUNTIME_H
#define WASM_RUNTIME_H

#include <stdbool.h>
#include <stdint.h>

/**
 * Initialize the WebAssembly runtime and load the URL filter module
 * 
 * @param prefix The URL prefix to filter (can be NULL for default)
 * @return 0 on success, -1 on failure
 */
int initialize(const char *prefix);

/**
 * Filter a URL based on the loaded WebAssembly module logic
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

#endif /* WASM_RUNTIME_H */ 