#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

// Define __WASM_EXPORT macro for exporting functions from WebAssembly
#define __WASM_EXPORT __attribute__((visibility("default")))

// Global variables to store statistics
static uint64_t accepted_count = 0;
static uint64_t rejected_count = 0;

// URL prefix to filter (default to "/aaaa")
static char url_prefix[256] = "/aaaa";

/**
 * Initialize the URL filter with a given prefix
 * 
 * @param prefix The URL prefix to filter. If NULL, keeps the default "/aaaa"
 * @return 0 on success
 */
__WASM_EXPORT int initialize(const char *prefix) {
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
    
    return 0;
}

/**
 * Filter a URL based on the configured prefix
 * 
 * @param url The URL to filter
 * @return true if the URL starts with the configured prefix, false otherwise
 */
__WASM_EXPORT bool url_filter(const char *url) {
    // Consider empty URLs as not matching
    if (url == NULL || *url == '\0') {
        rejected_count++;
        return false;
    }

    printf("url_prefix: %s\n", url_prefix);
    
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
 * Get the current counters for accepted and rejected URLs
 * 
 * @param accepted Pointer to store the number of accepted URLs
 * @param rejected Pointer to store the number of rejected URLs
 */
__WASM_EXPORT void get_counters(uint64_t *accepted, uint64_t *rejected) {
    if (accepted != NULL) {
        *accepted = accepted_count;
    }
    if (rejected != NULL) {
        *rejected = rejected_count;
    }
}

// Add explicit exports for Emscripten
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
EMSCRIPTEN_KEEPALIVE int module_initialize(const char* prefix) {
    return initialize(prefix);
}
EMSCRIPTEN_KEEPALIVE bool module_url_filter(const char* url) {
    return url_filter(url);
}
EMSCRIPTEN_KEEPALIVE void module_get_counters(uint64_t* accepted, uint64_t* rejected) {
    get_counters(accepted, rejected);
}
#endif 