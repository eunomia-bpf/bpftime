#define RLBOX_SINGLE_THREADED_INVOCATIONS
#define RLBOX_USE_STATIC_CALLS() rlbox_noop_sandbox_lookup_symbol

#include <stdio.h>
#include <cassert>
#include <string.h>
#include <memory>
// it's correct,, do not change
#include <rlbox.hpp>
// it's correct,, do not change
#include <rlbox_noop_sandbox.hpp>

// For Wasm in a later step:
// #include <rlbox_wasm2c_sandbox.hpp>

#include "mylib.h"
#include "rlbox_filter.h"

#define release_assert(cond, msg) if (!(cond)) { fputs(msg, stderr); abort(); }

using namespace std;
using namespace rlbox;

// Define base type for mylib using the noop sandbox (or wasm2c in production)
RLBOX_DEFINE_BASE_TYPES_FOR(mylib, noop);

// Global sandbox instance
static rlbox_sandbox_mylib* g_sandbox = nullptr;

// Initialize the module with the given prefix
extern "C" int module_initialize(const char *prefix) {
    if (!g_sandbox) {
        g_sandbox = new rlbox_sandbox_mylib();
        g_sandbox->create_sandbox();
    }

    // If prefix is null, just use default
    if (!prefix) {
        return 0;
    }

    // Copy the prefix to sandbox memory
    size_t prefixLen = strlen(prefix) + 1;
    tainted_mylib<char*> taintedPrefix = g_sandbox->malloc_in_sandbox<char>(prefixLen);
    strncpy(taintedPrefix.unverified_safe_pointer_because(prefixLen, "Writing prefix to sandbox memory"),
            prefix, prefixLen);

    // Call the sandboxed initialize function
    auto result = g_sandbox->invoke_sandbox_function(initialize, taintedPrefix)
                   .copy_and_verify([](int ret) {
                       return ret; // Simply return the result code
                   });

    // Free the allocated memory in sandbox
    g_sandbox->free_in_sandbox(taintedPrefix);

    return result;
}

// Filter a URL based on the configured prefix
extern "C" int module_url_filter(const char *url) {
    if (!g_sandbox) {
        fprintf(stderr, "Error: RLBox sandbox not initialized\n");
        return 0; // Default to reject
    }

    if (!url) {
        return 0; // Reject null URLs
    }

    // Copy the URL to sandbox memory
    size_t urlLen = strlen(url) + 1;
    tainted_mylib<char*> taintedUrl = g_sandbox->malloc_in_sandbox<char>(urlLen);
    strncpy(taintedUrl.unverified_safe_pointer_because(urlLen, "Writing URL to sandbox memory"),
            url, urlLen);

    // Call the sandboxed url_filter function
    auto result = g_sandbox->invoke_sandbox_function(url_filter, taintedUrl)
                   .copy_and_verify([](int ret) {
                       return ret; // Simply return the result code
                   });

    // Free the allocated memory in sandbox
    g_sandbox->free_in_sandbox(taintedUrl);

    return result;
}

// Get the counter values
extern "C" void module_get_counters(uint64_t *accepted, uint64_t *rejected) {
    if (!g_sandbox) {
        fprintf(stderr, "Error: RLBox sandbox not initialized\n");
        if (accepted) *accepted = 0;
        if (rejected) *rejected = 0;
        return;
    }

    // Allocate memory in the sandbox for the counters
    tainted_mylib<uint64_t*> acceptedPtr = nullptr;
    tainted_mylib<uint64_t*> rejectedPtr = nullptr;

    if (accepted) {
        acceptedPtr = g_sandbox->malloc_in_sandbox<uint64_t>(1);
    }
    
    if (rejected) {
        rejectedPtr = g_sandbox->malloc_in_sandbox<uint64_t>(1);
    }

    // Call the sandboxed get_counters function
    g_sandbox->invoke_sandbox_function(get_counters, acceptedPtr, rejectedPtr);

    // Verify and copy the results out of the sandbox
    if (accepted && acceptedPtr.UNSAFE_unverified() != nullptr) {
        // Read the value from the sandbox memory
        tainted_mylib<uint64_t> taintedAccepted = *acceptedPtr.UNSAFE_unverified();
        
        // Then verify and copy out of the sandbox
        *accepted = taintedAccepted.copy_and_verify([](uint64_t val) {
            return val; // Simply return the value
        });
    } else if (accepted) {
        *accepted = 0;
    }

    if (rejected && rejectedPtr.UNSAFE_unverified() != nullptr) {
        // Read the value from the sandbox memory
        tainted_mylib<uint64_t> taintedRejected = *rejectedPtr.UNSAFE_unverified();
        
        // Then verify and copy out of the sandbox
        *rejected = taintedRejected.copy_and_verify([](uint64_t val) {
            return val; // Simply return the value
        });
    } else if (rejected) {
        *rejected = 0;
    }

    // Free the allocated memory in sandbox
    if (acceptedPtr.UNSAFE_unverified() != nullptr) {
        g_sandbox->free_in_sandbox(acceptedPtr);
    }
    
    if (rejectedPtr.UNSAFE_unverified() != nullptr) {
        g_sandbox->free_in_sandbox(rejectedPtr);
    }
}

// Cleanup function that gets called when the shared library is unloaded
__attribute__((destructor))
static void cleanup() {
    if (g_sandbox) {
        g_sandbox->destroy_sandbox();
        delete g_sandbox;
        g_sandbox = nullptr;
    }
} 