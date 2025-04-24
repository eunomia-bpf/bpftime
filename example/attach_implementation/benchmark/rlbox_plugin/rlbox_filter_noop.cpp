#include <stdio.h>
#include <string.h>
#include <cstdint>
#include "rlbox_filter.h"

static const char* prefix = nullptr;
static uint64_t accepted = 0;
static uint64_t rejected = 0;

// Initialize the module with the given prefix
extern "C" int module_initialize(const char *_prefix) {
    if (_prefix) {
        size_t len = strlen(_prefix) + 1;
        char* new_prefix = new char[len];
        strncpy(new_prefix, _prefix, len);
        
        if (prefix) {
            delete[] prefix;
        }
        prefix = new_prefix;
    }
    
    return 0;
}

// Filter a URL based on the configured prefix
extern "C" int module_url_filter(const char *url) {
    if (!url) {
        rejected++;
        return 0; // Reject null URLs
    }
    
    if (!prefix || strncmp(url, prefix, strlen(prefix)) == 0) {
        accepted++;
        return 1; // URL starts with prefix or no prefix set, so accept
    } else {
        rejected++;
        return 0; // URL doesn't match prefix, reject
    }
}

// Get the counter values
extern "C" void module_get_counters(uint64_t *_accepted, uint64_t *_rejected) {
    if (_accepted) {
        *_accepted = accepted;
    }
    
    if (_rejected) {
        *_rejected = rejected;
    }
}

// Cleanup function that gets called when the shared library is unloaded
__attribute__((destructor))
static void cleanup() {
    if (prefix) {
        delete[] prefix;
        prefix = nullptr;
    }
} 