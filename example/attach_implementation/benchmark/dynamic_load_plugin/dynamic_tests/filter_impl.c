/*
 * SECURITY CONSIDERATIONS FOR ISOLATION:
 * 
 * This file represents a third-party plugin that will be called by the host (e.g., Nginx).
 * When isolating this plugin using technologies like WebAssembly, eBPF, or other sandboxing:
 * 
 * 1. GLOBAL STATE PROTECTION:
 *    - All global variables (accept_url_prefix, counter) must be isolated within the sandbox
 *    - Host should not have direct access to modify these values except through the API
 * 
 * 2. MEMORY BOUNDARY PROTECTION:
 *    - All memory used by the plugin should be isolated from the host's memory
 *    - String operations should be contained within the sandbox memory space
 * 
 * 3. API BOUNDARY PROTECTION:
 *    - Only explicitly exported functions should be callable by the host
 *    - All data passing through API boundaries requires validation
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>

// Configuration
// SECURITY: This global variable must be isolated within the sandbox memory
static char accept_url_prefix[128] = "/";

// Statistics
typedef struct {
    uint64_t accepted;
    uint64_t rejected;
} request_counter_t;

// SECURITY: This counter structure must be isolated within the sandbox memory
static request_counter_t counter = {0, 0};

// String comparison helper function (similar to the other implementations)
// SECURITY: String operations must operate only on sandbox memory
static int str_startswith(const char *str, const char *prefix) {
    size_t len_prefix = strlen(prefix);
    size_t len_str = strlen(str);
    
    if (len_str < len_prefix) {
        return 0;
    }
    
    return strncmp(str, prefix, len_prefix) == 0;
}

// The public API functions that will be dynamically loaded by the Nginx module
#ifdef __cplusplus
extern "C" {
#endif

// Initialize the module with the given prefix
// SECURITY: Host must validate 'prefix' and copy it to sandbox memory before calling
int module_initialize(const char *prefix) {
    if (prefix == NULL) {
        return -1;
    }
    
    // SECURITY: This string copy operation must be contained within sandbox memory
    strncpy(accept_url_prefix, prefix, sizeof(accept_url_prefix) - 1);
    accept_url_prefix[sizeof(accept_url_prefix) - 1] = '\0';
    
    // Reset counters
    counter.accepted = 0;
    counter.rejected = 0;
    
    printf("Filter implementation initialized with prefix: %s\n", accept_url_prefix);
    
    return 0;
}

// Main URL filtering function
// SECURITY: Host must validate 'url' and copy it to sandbox memory before calling
int module_url_filter(const char *url) {
    // Check if the URL starts with the accept prefix
    int result = str_startswith(url, accept_url_prefix);
    
    if (result) {
        counter.accepted++;
        return 1; // Allow
    } else {
        counter.rejected++;
        return 0; // Block
    }
}

// Get the counter values
// SECURITY: Host must provide pointers to sandbox memory regions
// After call, host should copy values from sandbox memory to host memory
void module_get_counters(uint64_t *accepted, uint64_t *rejected) {
    if (accepted) *accepted = counter.accepted;
    if (rejected) *rejected = counter.rejected;
}

#ifdef __cplusplus
}
#endif 