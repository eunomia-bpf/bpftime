#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

// For WASI compatibility, we don't use __WASM_EXPORT macro
// The exports are handled by linker flags in the Makefile

// Global variables to store statistics
static uint64_t accepted_count = 0;
static uint64_t rejected_count = 0;

// URL prefix to filter (default to "/aaaa")
static char url_prefix[256] = "/aaaa";

// Add a buffer for data exchange between host and WASM
static char data_buffer[1024] = {0};
static size_t buffer_size = 0;

/**
 * Initialize the URL filter with a given prefix
 * 
 * @param prefix The URL prefix to filter. If NULL, keeps the default "/aaaa"
 * @return 0 on success
 */
int initialize(const char *prefix) {
    // If prefix is provided, update the filter prefix
    if (prefix != NULL && *prefix != '\0') {
        size_t len = strlen(prefix);
        if (len >= sizeof(url_prefix)) {
            len = sizeof(url_prefix) - 1;
        }
        memcpy(url_prefix, prefix, len);
        url_prefix[len] = '\0';
    }
    
    // Reset counters
    accepted_count = 0;
    rejected_count = 0;
    
    // Reset buffer
    memset(data_buffer, 0, sizeof(data_buffer));
    buffer_size = 0;
    
    return 0;
}

/**
 * Filter a URL based on the configured prefix
 * 
 * @param url The URL to filter
 * @return true if the URL starts with the configured prefix, false otherwise
 */
bool url_filter(const char *url) {
    // Consider empty URLs as not matching
    if (url == NULL || *url == '\0') {
        rejected_count++;
        return false;
    }
    
    // Get URL prefix length
    size_t prefix_len = strlen(url_prefix);
    
    // Check if URL starts with the prefix
    bool matches = (strncmp(url, url_prefix, prefix_len) == 0);
    
    // Update counters
    if (matches) {
        accepted_count++;
    } else {
        rejected_count++;
    }
    
    return matches;
}

/**
 * Set the data in the buffer
 *
 * @param data The data to copy into the buffer
 * @param size Size of the data to copy
 * @return 0 on success, -1 if buffer size exceeded
 */
int set_buffer(const char *data, size_t size) {
    if (size >= sizeof(data_buffer)) {
        return -1;  // Buffer overflow
    }
    
    memcpy(data_buffer, data, size);
    buffer_size = size;
    data_buffer[size] = '\0';  // Ensure null-termination
    
    return 0;
}

/**
 * Get the data from the buffer
 *
 * @param out_data Pointer to store buffer data pointer
 * @param out_size Pointer to store buffer size
 */
void get_buffer(const char **out_data, size_t *out_size) {
    if (out_data != NULL) {
        *out_data = data_buffer;
    }
    
    if (out_size != NULL) {
        *out_size = buffer_size;
    }
}

/**
 * Get the current counters for accepted and rejected URLs
 * 
 * @param accepted Pointer to store the number of accepted URLs
 * @param rejected Pointer to store the number of rejected URLs
 */
void get_counters(uint64_t *accepted, uint64_t *rejected) {
    if (accepted != NULL) {
        *accepted = accepted_count;
    }
    if (rejected != NULL) {
        *rejected = rejected_count;
    }
}

// No Emscripten-specific code is needed as we're using WASI
// The exports are handled by the linker flags in the Makefile 