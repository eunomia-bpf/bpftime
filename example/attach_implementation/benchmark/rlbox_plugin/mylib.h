#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize the URL filter module with a prefix
 * 
 * @param prefix The URL prefix to accept
 * @return 0 on success, -1 on failure
 */
int initialize(const char *prefix);

/**
 * Filter a URL based on the configured prefix
 * 
 * @param url The URL to filter
 * @return 1 if the URL is accepted, 0 if rejected
 */
int url_filter(const char *url);

/**
 * Get the counter values for accepted and rejected requests
 * 
 * @param accepted Pointer to store the number of accepted requests
 * @param rejected Pointer to store the number of rejected requests
 */
void get_counters(uint64_t *accepted, uint64_t *rejected);

#ifdef __cplusplus
}
#endif 