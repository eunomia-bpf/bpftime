#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <dlfcn.h>

#include "wasm3.h"
#include "m3_env.h"

// Our exported interface
bool url_filter(const char *url);
int initialize(const char *prefix);
void get_counters(uint64_t *accepted, uint64_t *rejected);

// Exported functions for the dynamic loader with the module_ prefix
int module_initialize(const char *prefix) {
    return initialize(prefix);
}

bool module_url_filter(const char *url) {
    // For the dynamic loader, we need to return the OPPOSITE of our internal function
    // The dynamic loader expects:
    // - Return 1 for URLs that should be ALLOWED (HTTP 200)
    // - Return 0 for URLs that should be REJECTED (HTTP 403)
    
    // Our internal url_filter returns:
    // - true if URL passes the filter
    // - false if URL fails the filter
    
    // So we need to convert between the two conventions
    bool internal_result = url_filter(url);
    
    // Return 1 (true) to allow the URL (HTTP 200)
    // Return 0 (false) to reject the URL (HTTP 403)
    // This matches what the dynamic_load_module expects
    fprintf(stderr, "Dynamic loader filter for URL %s: internal_result=%d, returning=%d\n", 
            url, internal_result, internal_result ? 1 : 0);
            
    return internal_result ? 1 : 0;
}

void module_get_counters(uint64_t *accepted, uint64_t *rejected) {
    get_counters(accepted, rejected);
}

// WebAssembly runtime state
static IM3Environment env = NULL;
static IM3Runtime runtime = NULL;
static IM3Module module = NULL;

// Function pointers to the WebAssembly functions
static IM3Function initialize_func = NULL;
static IM3Function url_filter_func = NULL;
static IM3Function get_counters_func = NULL;

// Statistics counters (in case WASM module fails)
static uint64_t local_accepted = 0;
static uint64_t local_rejected = 0;

// Helper for error handling
#define CHECK_WASM_ERROR(call, msg) \
    { \
        M3Result result = call; \
        if (result != m3Err_none) { \
            fprintf(stderr, "WASM error: %s: %s\n", msg, result); \
            return false; \
        } \
    }

// Initialize WASM runtime and load the module
int initialize(const char *prefix) {
    M3Result result;
    
    // Get the path to the WASM module from environment variable
    const char *wasm_module_path = getenv("WASM_MODULE_PATH");
    if (wasm_module_path == NULL) {
        fprintf(stderr, "WASM_MODULE_PATH environment variable is not set\n");
        return -1;
    }
    
    fprintf(stderr, "Loading WebAssembly module from: %s\n", wasm_module_path);
    
    // Read WASM file
    FILE *f = fopen(wasm_module_path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open WebAssembly module: %s\n", wasm_module_path);
        return -1;
    }
    
    // Get file size
    fseek(f, 0, SEEK_END);
    size_t wasm_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    fprintf(stderr, "WebAssembly module size: %zu bytes\n", wasm_size);
    
    // Allocate memory and read the file
    unsigned char *wasm_buffer = malloc(wasm_size);
    if (!wasm_buffer) {
        fclose(f);
        fprintf(stderr, "Out of memory\n");
        return -1;
    }
    
    if (fread(wasm_buffer, 1, wasm_size, f) != wasm_size) {
        free(wasm_buffer);
        fclose(f);
        fprintf(stderr, "Failed to read WebAssembly module\n");
        return -1;
    }
    
    fclose(f);
    
    // Initialize Wasm3
    env = m3_NewEnvironment();
    if (!env) {
        free(wasm_buffer);
        fprintf(stderr, "Failed to create Wasm3 environment\n");
        return -1;
    }
    
    // Create runtime with more memory
    runtime = m3_NewRuntime(env, 128 * 1024, NULL);
    if (!runtime) {
        free(wasm_buffer);
        m3_FreeEnvironment(env);
        fprintf(stderr, "Failed to create Wasm3 runtime\n");
        return -1;
    }
    
    result = m3_ParseModule(env, &module, wasm_buffer, wasm_size);
    free(wasm_buffer);
    
    if (result) {
        m3_FreeRuntime(runtime);
        m3_FreeEnvironment(env);
        fprintf(stderr, "Failed to parse WebAssembly module: %s\n", result);
        return -1;
    }
    
    result = m3_LoadModule(runtime, module);
    if (result) {
        m3_FreeModule(module);
        m3_FreeRuntime(runtime);
        m3_FreeEnvironment(env);
        fprintf(stderr, "Failed to load WebAssembly module: %s\n", result);
        return -1;
    }
    
    // Link functions
    result = m3_FindFunction(&initialize_func, runtime, "initialize");
    if (result) {
        fprintf(stderr, "Failed to find initialize function: %s\n", result);
        return -1;
    }
    
    result = m3_FindFunction(&url_filter_func, runtime, "url_filter");
    if (result) {
        fprintf(stderr, "Failed to find url_filter function: %s\n", result);
        return -1;
    }
    
    result = m3_FindFunction(&get_counters_func, runtime, "get_counters");
    if (result) {
        fprintf(stderr, "Failed to find get_counters function: %s\n", result);
        return -1;
    }
    
    // Call initialize function with the prefix
    if (prefix != NULL) {
        // Create a copy of the string to ensure it's not garbage collected
        const void* args[1] = { prefix };
        result = m3_CallArgv(initialize_func, 1, args);
        if (result) {
            fprintf(stderr, "Failed to call initialize function: %s\n", result);
            return -1;
        }
    }
    
    fprintf(stderr, "WebAssembly URL filter initialized successfully\n");
    return 0;
}

// The URL filter function
bool url_filter(const char *url) {
    if (!url_filter_func) {
        fprintf(stderr, "WebAssembly runtime not initialized\n");
        local_rejected++;
        return false;
    }
    
    fprintf(stderr, "Filtering URL: %s\n", url);
    
    const void* args[1] = { url };
    M3Result result = m3_CallArgv(url_filter_func, 1, args);
    
    if (result) {
        fprintf(stderr, "Failed to call url_filter function: %s\n", result);
        local_rejected++;
        return false;
    }
    
    // Get return value (bool)
    uint32_t ret = 0;
    result = m3_GetResultsV(url_filter_func, &ret);
    if (result) {
        fprintf(stderr, "Failed to get result from url_filter function: %s\n", result);
        local_rejected++;
        return false;
    }
    
    // Update local counters
    if (ret) {
        local_accepted++;
        fprintf(stderr, "URL accepted: %s (ret=%u)\n", url, ret);
    } else {
        local_rejected++;
        fprintf(stderr, "URL rejected: %s (ret=%u)\n", url, ret);
    }
    
    // IMPORTANT: In our API true means ALLOW the URL (return 200)
    // But in the NGINX filter context, returning true means the URL matches the filter, 
    // which causes NGINX to continue processing the request
    return (bool)ret;
}

// Get statistics counters
void get_counters(uint64_t *accepted, uint64_t *rejected) {
    if (!get_counters_func) {
        fprintf(stderr, "WebAssembly runtime not initialized\n");
        if (accepted) *accepted = local_accepted;
        if (rejected) *rejected = local_rejected;
        return;
    }
    
    // Prepare pointers to pass to the WebAssembly function
    void* args[2] = { accepted, rejected };
    M3Result result = m3_CallArgv(get_counters_func, 2, args);
    
    if (result) {
        fprintf(stderr, "Failed to call get_counters function: %s\n", result);
        if (accepted) *accepted = local_accepted;
        if (rejected) *rejected = local_rejected;
    }
}

// Cleanup function
__attribute__((destructor))
static void cleanup() {
    if (module) {
        m3_FreeModule(module);
        module = NULL;
    }
    
    if (runtime) {
        m3_FreeRuntime(runtime);
        runtime = NULL;
    }
    
    if (env) {
        m3_FreeEnvironment(env);
        env = NULL;
    }
    
    initialize_func = NULL;
    url_filter_func = NULL;
    get_counters_func = NULL;
} 