#define RLBOX_SINGLE_THREADED_INVOCATIONS
// Use dynamic loading for wasm2c
#define RLBOX_USE_DYNAMIC_CALLS() rlbox_wasm2c_sandbox_lookup_symbol

#include <stdio.h>
#include <cassert>
#include <string.h>
#include <memory>
#include <rlbox.hpp>
#include <rlbox_wasm2c_sandbox.hpp>

#include "mylib.h"
#include "rlbox_filter.h"

#define release_assert(cond, msg) if (!(cond)) { fputs(msg, stderr); abort(); }

using namespace std;
using namespace rlbox;

// Define base type for mylib using the wasm2c sandbox
RLBOX_DEFINE_BASE_TYPES_FOR(mylib, wasm2c);

// Global sandbox instance
static rlbox_sandbox_mylib* g_sandbox = nullptr;

// Path to the compiled WASM module
static const char* WASM_PATH = "mylib.wasm";

// Initialize the module with the given prefix
extern "C" int module_initialize(const char *prefix) {
    if (!g_sandbox) {
        // Check for environment variable to override WASM path
        const char* env_wasm_path = getenv("RLBOX_WASM_PATH");
        const char* wasm_path = env_wasm_path ? env_wasm_path : WASM_PATH;
        
        fprintf(stderr, "Creating wasm2c sandbox with module: %s\n", wasm_path);
        
        g_sandbox = new rlbox_sandbox_mylib();
        g_sandbox->create_sandbox(wasm_path);
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

    // Create variables to hold the results
    tainted_mylib<uint64_t> acceptedVal = 0;
    tainted_mylib<uint64_t> rejectedVal = 0;
    
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
    if (accepted && acceptedPtr) {
        // Create a verification function to safely copy data from the sandbox
        auto verifier = [](tainted_mylib<uint64_t> val) -> uint64_t {
            // Simplest verification: just return the value
            return val;
        };
        
        // Create a temporary variable to hold the value from the sandbox
        tainted_mylib<uint64_t> taintedAccepted = acceptedPtr.UNSAFE_unverified()[0];
        *accepted = taintedAccepted.copy_and_verify(verifier);
    }

    if (rejected && rejectedPtr) {
        auto verifier = [](tainted_mylib<uint64_t> val) -> uint64_t {
            // Simplest verification: just return the value
            return val;
        };

        tainted_mylib<uint64_t> taintedRejected = rejectedPtr.UNSAFE_unverified()[0];
        *rejected = taintedRejected.copy_and_verify(verifier);
    }

    // Free the allocated memory in sandbox
    if (acceptedPtr) g_sandbox->free_in_sandbox(acceptedPtr);
    if (rejectedPtr) g_sandbox->free_in_sandbox(rejectedPtr);
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