/*
 * SECURITY CONSIDERATIONS FOR ISOLATION:
 * 
 * This header defines the API boundary between the host (e.g., Nginx) and the
 * third-party plugin. When using isolation technologies like WebAssembly, eBPF,
 * or other sandboxing mechanisms:
 * 
 * 1. API BOUNDARY CONSIDERATIONS:
 *    - These functions form the ONLY permitted interface between host and plugin
 *    - No other functions should be directly callable from outside the sandbox
 * 
 * 2. DATA CROSSING BOUNDARIES:
 *    - All parameters passed between host and plugin must be validated
 *    - String data should be copied between host and sandbox memory spaces
 *    - Pointers should reference memory within the appropriate memory space
 * 
 * 3. HOST RESPONSIBILITIES:
 *    - Host must copy input strings to sandbox memory before calling functions
 *    - Host must provide pointers to sandbox memory for outputs
 *    - Host must copy returned data from sandbox memory to host memory after calls
 *    - Host should validate all return values before using them
 */

#ifndef FILTER_IMPL_H
#define FILTER_IMPL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize the URL filter module with a prefix
 * 
 * SECURITY: 
 * - Host must validate and copy 'prefix' to sandbox memory before calling
 * - String operations inside function must be contained within sandbox memory
 * 
 * @param prefix The URL prefix to accept
 * @return 0 on success, -1 on failure
 */
int module_initialize(const char *prefix);

/**
 * Filter a URL based on the configured prefix
 * 
 * SECURITY:
 * - Host must validate and copy 'url' to sandbox memory before calling
 * - String comparison must occur inside sandbox memory
 * - Return value is a simple integer, safe to pass directly to host
 * 
 * @param url The URL to filter
 * @return 1 if the URL is accepted, 0 if rejected
 */
int module_url_filter(const char *url);

/**
 * Get the counter values for accepted and rejected requests
 * 
 * SECURITY:
 * - Host must provide pointers to sandbox memory regions
 * - After call, host should copy values from sandbox memory to host memory
 * - Plugin must not dereference pointers outside its memory space
 * 
 * @param accepted Pointer to store the number of accepted requests
 * @param rejected Pointer to store the number of rejected requests
 */
void module_get_counters(uint64_t *accepted, uint64_t *rejected);

#ifdef __cplusplus
}
#endif

#endif /* FILTER_IMPL_H */ 