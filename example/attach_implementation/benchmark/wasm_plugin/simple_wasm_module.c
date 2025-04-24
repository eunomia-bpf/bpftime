#include <string.h>
#include <stdint.h>

// Fixed memory for the prefix string
static char accept_url_prefix[128] = "/";

// Statistics
static uint64_t accepted_count = 0;
static uint64_t rejected_count = 0;

// String comparison helper function (same as in other implementations)
static int str_startswith(const char *str, const char *prefix) {
    if (!str || !prefix) {
        return 0;
    }
    
    size_t len_prefix = strlen(prefix);
    size_t len_str = strlen(str);
    
    if (len_str < len_prefix) {
        return 0;
    }
    
    return strncmp(str, prefix, len_prefix) == 0;
}

// Initialize the WebAssembly module with the given prefix
// We'll just store the prefix directly, without any dynamic memory
int module_initialize(const char *prefix) {
    if (!prefix) {
        // Default prefix if none provided
        strcpy(accept_url_prefix, "/");
    } else {
        strncpy(accept_url_prefix, prefix, sizeof(accept_url_prefix) - 1);
        accept_url_prefix[sizeof(accept_url_prefix) - 1] = '\0';
    }
    
    // Reset counters
    accepted_count = 0;
    rejected_count = 0;
    
    return 0;
}

// Main URL filtering function
int module_url_filter(const char *url) {
    if (!url) {
        rejected_count++;
        return 0; // Reject NULL URLs
    }
    
    // Check if the URL starts with the accept prefix
    int result = str_startswith(url, accept_url_prefix);
    
    if (result) {
        accepted_count++;
        return 1; // Allow
    } else {
        rejected_count++;
        return 0; // Block
    }
}

// Get the counter values
void module_get_counters(uint64_t *accepted, uint64_t *rejected) {
    if (accepted) *accepted = accepted_count;
    if (rejected) *rejected = rejected_count;
}

// Simple memory implementation for WebAssembly
void* malloc(size_t size) {
    // Not implemented - we don't need dynamic memory in this example
    return 0;
}

void free(void* ptr) {
    // Not implemented
} 