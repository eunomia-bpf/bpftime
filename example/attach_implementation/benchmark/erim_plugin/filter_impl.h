/*
 * SECURITY IMPLEMENTATION WITH ERIM:
 * 
 * This header defines the API for a URL filtering plugin protected by ERIM 
 * (Efficient Remote Isolation with Memory Protection Keys). ERIM provides:
 * 
 * 1. HARDWARE-ENFORCED ISOLATION:
 *    - Uses Intel MPK to create separate trusted/untrusted memory domains
 *    - Protects plugin state from unauthorized access by the host
 * 
 * 2. API BOUNDARY PROTECTION:
 *    - Each function properly switches between trusted/untrusted domains
 *    - Prevents direct memory access from outside the trusted domain
 * 
 * 3. INTEGRITY PROTECTION:
 *    - Ensures global state cannot be modified by untrusted code
 */

#ifndef FILTER_IMPL_H
#define FILTER_IMPL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Request counter structure - exposed for testing purposes only */
typedef struct {
    uint64_t accepted;
    uint64_t rejected;
} request_counter_t;

/* This is marked as extern for testing ERIM protection */
extern request_counter_t counter;

/**
 * Initialize the URL filter module with a prefix
 * 
 * SECURITY:
 * - Sets up ERIM protection on first call
 * - Switches to trusted domain to securely store the prefix
 * - Prevents modification of internal state outside this function
 * 
 * @param prefix The URL prefix to accept
 * @return 0 on success, -1 on failure
 */
int module_initialize(const char *prefix);

/**
 * Filter a URL based on the configured prefix
 * 
 * SECURITY:
 * - Switches to trusted domain to access the protected prefix and counters
 * - All string operations occur in the trusted domain
 * - Access to the counter variables is protected
 * 
 * @param url The URL to filter
 * @return 1 if the URL is accepted, 0 if rejected
 */
int module_url_filter(const char *url);

/**
 * Get the counter values for accepted and rejected requests
 * 
 * SECURITY:
 * - Switches to trusted domain to access protected counter data
 * - Prevents modification of counters by untrusted code
 * - Exposes only the specific counter values, not the entire state
 * 
 * @param accepted Pointer to store the number of accepted requests
 * @param rejected Pointer to store the number of rejected requests
 */
void module_get_counters(uint64_t *accepted, uint64_t *rejected);

#ifdef __cplusplus
}
#endif

#endif /* FILTER_IMPL_H */ 