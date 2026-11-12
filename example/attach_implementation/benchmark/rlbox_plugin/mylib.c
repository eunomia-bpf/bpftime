#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "mylib.h"

// Configuration
static char accept_url_prefix[128] = "/";

// Statistics
typedef struct {
    uint64_t accepted;
    uint64_t rejected;
} request_counter_t;

static request_counter_t counter = {0, 0};

// String comparison helper function
static int str_startswith(const char *str, const char *prefix) {
    size_t len_prefix = strlen(prefix);
    size_t len_str = strlen(str);
    
    if (len_str < len_prefix) {
        return 0;
    }
    
    return strncmp(str, prefix, len_prefix) == 0;
}

// Initialize the module with the given prefix
int initialize(const char *prefix) {
    if (prefix == NULL) {
        return -1;
    }
    
    strncpy(accept_url_prefix, prefix, sizeof(accept_url_prefix) - 1);
    accept_url_prefix[sizeof(accept_url_prefix) - 1] = '\0';
    
    // Reset counters
    counter.accepted = 0;
    counter.rejected = 0;
    
    printf("Filter implementation initialized with prefix: %s\n", accept_url_prefix);
    
    return 0;
}

// Main URL filtering function
int url_filter(const char *url) {
    if (url == NULL) {
        counter.rejected++;
        return 0; // Reject NULL URLs
    }
    
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
void get_counters(uint64_t *accepted, uint64_t *rejected) {
    if (accepted) *accepted = counter.accepted;
    if (rejected) *rejected = counter.rejected;
} 