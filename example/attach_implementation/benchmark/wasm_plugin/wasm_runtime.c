#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <dlfcn.h>

/* WAMR headers */
#include "wasm_export.h"

// Our exported interface
bool url_filter(const char *url);
int initialize(const char *prefix);
void get_counters(uint64_t *accepted, uint64_t *rejected);
int set_buffer(const char *data, size_t size);
int get_buffer(const char **out_data, size_t *out_size);

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

int module_set_buffer(const char *data, size_t size) {
    return set_buffer(data, size);
}

int module_get_buffer(const char **out_data, size_t *out_size) {
    return get_buffer(out_data, out_size);
}

// WAMR runtime state
static wasm_module_t module = NULL;
static wasm_module_inst_t module_inst = NULL;
static wasm_exec_env_t exec_env = NULL;

// Function names from WASM module
static const char *INITIALIZE_FUNC = "initialize";
static const char *URL_FILTER_FUNC = "url_filter";
static const char *GET_COUNTERS_FUNC = "get_counters";
static const char *SET_BUFFER_FUNC = "set_buffer";
static const char *GET_BUFFER_FUNC = "get_buffer";

// Helper for error handling
#define CHECK_ERROR(cond, msg) \
    if (!(cond)) { \
        fprintf(stderr, "WASM error: %s\n", msg); \
        return -1; \
    }

// Dummy implementation for any required host functions
static int dummy_printf(wasm_exec_env_t exec_env, const char *format, ...) {
    (void)exec_env;
    (void)format;
    return 0;
}

static int dummy_puts(wasm_exec_env_t exec_env, const char *str) {
    (void)exec_env;
    (void)str;
    return 0;
}

static void dummy_abort(wasm_exec_env_t exec_env) {
    (void)exec_env;
    fprintf(stderr, "WASM abort called\n");
}

static int dummy_main(wasm_exec_env_t exec_env, int argc, char **argv) {
    (void)exec_env;
    (void)argc;
    (void)argv;
    return 0;
}

// Register native functions that the WASM module might need
static NativeSymbol native_symbols[] = {
    {"printf", dummy_printf, "(*)i", NULL},
    {"puts", dummy_puts, "($)i", NULL},
    {"abort", dummy_abort, "()v", NULL},
    {"__main_argc_argv", dummy_main, "(ii)i", NULL},
};

// Initialize WAMR runtime and load the module
int initialize(const char *prefix) {
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
    
    // Initialize WAMR runtime
    // Initialize runtime environment with proper system allocator
    if (!wasm_runtime_init()) {
        fprintf(stderr, "Failed to initialize WAMR runtime\n");
        free(wasm_buffer);
        return -1;
    }
    
    // Register native symbols (for printf, etc.)
    if (!wasm_runtime_register_natives("env", native_symbols, 
                                     sizeof(native_symbols) / sizeof(NativeSymbol))) {
        fprintf(stderr, "Failed to register native functions\n");
        free(wasm_buffer);
        wasm_runtime_destroy();
        return -1;
    }
    
    // Load the WASM module
    char error_buf[128];
    module = wasm_runtime_load(wasm_buffer, (uint32_t)wasm_size, 
                              error_buf, sizeof(error_buf));
    free(wasm_buffer);
    
    if (!module) {
        fprintf(stderr, "Failed to load WASM module: %s\n", error_buf);
        wasm_runtime_destroy();
        return -1;
    }
    
    // Instantiate the module with more memory
    module_inst = wasm_runtime_instantiate(module, 
                                         64 * 1024,   // stack size 
                                         64 * 1024,   // heap size
                                         error_buf, sizeof(error_buf));
    
    if (!module_inst) {
        fprintf(stderr, "Failed to instantiate WASM module: %s\n", error_buf);
        wasm_runtime_unload(module);
        wasm_runtime_destroy();
        return -1;
    }
    
    // Create execution environment
    exec_env = wasm_runtime_create_exec_env(module_inst, 32 * 1024);
    if (!exec_env) {
        fprintf(stderr, "Failed to create execution environment\n");
        wasm_runtime_deinstantiate(module_inst);
        wasm_runtime_unload(module);
        wasm_runtime_destroy();
        return -1;
    }
    
    // Call initialize function with the prefix if provided
    if (prefix != NULL) {
        // Get function
        wasm_function_inst_t func = wasm_runtime_lookup_function(module_inst, INITIALIZE_FUNC);
        if (!func) {
            fprintf(stderr, "Failed to find initialize function\n");
            return -1;
        }
        
        // Allocate memory for prefix in WASM space
        uint32_t prefix_len = (uint32_t)strlen(prefix) + 1;
        wasm_module_inst_t inst = wasm_runtime_get_module_inst(exec_env);
        uint32_t prefix_offset = wasm_runtime_module_malloc(inst, prefix_len, NULL);
        if (!prefix_offset) {
            fprintf(stderr, "Failed to allocate memory for prefix\n");
            return -1;
        }
        
        // Copy string to WASM memory
        if (!wasm_runtime_validate_app_addr(inst, prefix_offset, prefix_len)) {
            fprintf(stderr, "Failed to validate app addr\n");
            wasm_runtime_module_free(inst, prefix_offset);
            return -1;
        }
        
        void *native_ptr = wasm_runtime_addr_app_to_native(inst, prefix_offset);
        memcpy(native_ptr, prefix, prefix_len);
        
        // Call function with the prefix parameter
        uint32_t argv[1] = { prefix_offset };
        if (!wasm_runtime_call_wasm(exec_env, func, 1, argv)) {
            const char *exception = wasm_runtime_get_exception(module_inst);
            fprintf(stderr, "Failed to call initialize function: %s\n", exception ? exception : "Unknown error");
            wasm_runtime_module_free(inst, prefix_offset);
            return -1;
        }
        
        // Free allocated memory
        wasm_runtime_module_free(inst, prefix_offset);
    }
    
    fprintf(stderr, "WebAssembly URL filter initialized successfully\n");
    return 0;
}

// The URL filter function
bool url_filter(const char *url) {
    if (!module_inst || !exec_env) {
        fprintf(stderr, "WebAssembly runtime not initialized\n");
        return false;
    }
    
    fprintf(stderr, "Filtering URL: %s\n", url);
    
    // Get function
    wasm_function_inst_t func = wasm_runtime_lookup_function(module_inst, URL_FILTER_FUNC);
    if (!func) {
        fprintf(stderr, "Failed to find url_filter function\n");
        return false;
    }
    
    // Allocate memory for URL in WASM space
    uint32_t url_len = (uint32_t)strlen(url) + 1;
    wasm_module_inst_t inst = wasm_runtime_get_module_inst(exec_env);
    uint32_t url_offset = wasm_runtime_module_malloc(inst, url_len, NULL);
    if (!url_offset) {
        fprintf(stderr, "Failed to allocate memory for URL\n");
        return false;
    }
    
    // Copy string to WASM memory
    if (!wasm_runtime_validate_app_addr(inst, url_offset, url_len)) {
        fprintf(stderr, "Failed to validate app addr\n");
        wasm_runtime_module_free(inst, url_offset);
        return false;
    }
    
    void *native_ptr = wasm_runtime_addr_app_to_native(inst, url_offset);
    memcpy(native_ptr, url, url_len);
    
    // Call function with the URL parameter
    uint32_t argv[1] = { url_offset };
    if (!wasm_runtime_call_wasm(exec_env, func, 1, argv)) {
        const char *exception = wasm_runtime_get_exception(module_inst);
        fprintf(stderr, "Failed to call url_filter function: %s\n", exception ? exception : "Unknown error");
        wasm_runtime_module_free(inst, url_offset);
        return false;
    }
    
    // Get return value
    uint32_t ret = argv[0];
    
    // Free allocated memory
    wasm_runtime_module_free(inst, url_offset);
   
    // IMPORTANT: In our API true means ALLOW the URL (return 200)
    // But in the NGINX filter context, returning true means the URL matches the filter,
    // which causes NGINX to continue processing the request
    return (bool)ret;
}

// Set data in the WebAssembly module's buffer
int set_buffer(const char *data, size_t size) {
    if (!module_inst || !exec_env) {
        fprintf(stderr, "WebAssembly runtime not initialized\n");
        return -1;
    }
    
    fprintf(stderr, "Setting buffer with %zu bytes of data\n", size);
    
    // Get function
    wasm_function_inst_t func = wasm_runtime_lookup_function(module_inst, SET_BUFFER_FUNC);
    if (!func) {
        fprintf(stderr, "Failed to find set_buffer function\n");
        return -1;
    }
    
    wasm_module_inst_t inst = wasm_runtime_get_module_inst(exec_env);
    
    // Allocate memory for data in WASM space
    uint32_t data_offset = wasm_runtime_module_malloc(inst, (uint32_t)size, NULL);
    if (!data_offset) {
        fprintf(stderr, "Failed to allocate memory for buffer data\n");
        return -1;
    }
    
    // Copy data to WASM memory
    if (!wasm_runtime_validate_app_addr(inst, data_offset, (uint32_t)size)) {
        fprintf(stderr, "Failed to validate app addr for data\n");
        wasm_runtime_module_free(inst, data_offset);
        return -1;
    }
    
    void *native_data_ptr = wasm_runtime_addr_app_to_native(inst, data_offset);
    memcpy(native_data_ptr, data, size);
    
    // Call function with data and size parameters
    uint32_t argv[2] = { data_offset, (uint32_t)size };
    if (!wasm_runtime_call_wasm(exec_env, func, 2, argv)) {
        const char *exception = wasm_runtime_get_exception(module_inst);
        fprintf(stderr, "Failed to call set_buffer function: %s\n", exception ? exception : "Unknown error");
        wasm_runtime_module_free(inst, data_offset);
        return -1;
    }
    
    // Get return value
    int32_t ret = (int32_t)argv[0];
    
    // Free allocated memory
    wasm_runtime_module_free(inst, data_offset);
    
    return ret;
}

// Get data from the WebAssembly module's buffer
int get_buffer(const char **out_data, size_t *out_size) {
    if (!module_inst || !exec_env) {
        fprintf(stderr, "WebAssembly runtime not initialized\n");
        return -1;
    }
    
    // Get function
    wasm_function_inst_t func = wasm_runtime_lookup_function(module_inst, GET_BUFFER_FUNC);
    if (!func) {
        fprintf(stderr, "Failed to find get_buffer function\n");
        return -1;
    }
    
    wasm_module_inst_t inst = wasm_runtime_get_module_inst(exec_env);
    
    // Allocate memory for output pointers in WASM space (two 32-bit values)
    uint32_t out_data_offset = wasm_runtime_module_malloc(inst, 4, NULL);
    if (!out_data_offset) {
        fprintf(stderr, "Failed to allocate memory for out_data pointer\n");
        return -1;
    }
    
    uint32_t out_size_offset = wasm_runtime_module_malloc(inst, 4, NULL);
    if (!out_size_offset) {
        fprintf(stderr, "Failed to allocate memory for out_size pointer\n");
        wasm_runtime_module_free(inst, out_data_offset);
        return -1;
    }
    
    // Call function with pointers to receive the values
    uint32_t argv[2] = { out_data_offset, out_size_offset };
    if (!wasm_runtime_call_wasm(exec_env, func, 2, argv)) {
        const char *exception = wasm_runtime_get_exception(module_inst);
        fprintf(stderr, "Failed to call get_buffer function: %s\n", exception ? exception : "Unknown error");
        wasm_runtime_module_free(inst, out_data_offset);
        wasm_runtime_module_free(inst, out_size_offset);
        return -1;
    }
    
    // Read the output values
    void *native_data_ptr = wasm_runtime_addr_app_to_native(inst, out_data_offset);
    void *native_size_ptr = wasm_runtime_addr_app_to_native(inst, out_size_offset);
    
    uint32_t buffer_offset = *(uint32_t*)native_data_ptr;
    uint32_t buffer_size = *(uint32_t*)native_size_ptr;
    
    // Get a pointer to the actual buffer data
    if (!wasm_runtime_validate_app_addr(inst, buffer_offset, buffer_size)) {
        fprintf(stderr, "Failed to validate buffer address\n");
        wasm_runtime_module_free(inst, out_data_offset);
        wasm_runtime_module_free(inst, out_size_offset);
        return -1;
    }
    
    // Set the output parameters
    *out_data = wasm_runtime_addr_app_to_native(inst, buffer_offset);
    *out_size = buffer_size;
    
    // Free allocated memory for the pointers (not the buffer itself, which is managed by WASM)
    wasm_runtime_module_free(inst, out_data_offset);
    wasm_runtime_module_free(inst, out_size_offset);
    
    fprintf(stderr, "Got buffer with %zu bytes of data\n", *out_size);
    return 0;
}

// Get statistics counters
void get_counters(uint64_t *accepted, uint64_t *rejected) {
    if (!module_inst || !exec_env) {
        fprintf(stderr, "WebAssembly runtime not initialized\n");
        return;
    }
    
    // Get function
    wasm_function_inst_t func = wasm_runtime_lookup_function(module_inst, GET_COUNTERS_FUNC);
    if (!func) {
        fprintf(stderr, "Failed to find get_counters function\n");
        return;
    }
    
    wasm_module_inst_t inst = wasm_runtime_get_module_inst(exec_env);
    
    // Allocate memory for output variables in WASM space (two 64-bit values)
    uint32_t accepted_offset = wasm_runtime_module_malloc(inst, 8, NULL);
    if (!accepted_offset) {
        fprintf(stderr, "Failed to allocate memory for accepted counter\n");
        return;
    }
    
    uint32_t rejected_offset = wasm_runtime_module_malloc(inst, 8, NULL);
    if (!rejected_offset) {
        fprintf(stderr, "Failed to allocate memory for rejected counter\n");
        wasm_runtime_module_free(inst, accepted_offset);
        return;
    }
    
    // Call function with pointers to receive the values
    uint32_t argv[2] = { accepted_offset, rejected_offset };
    if (!wasm_runtime_call_wasm(exec_env, func, 2, argv)) {
        const char *exception = wasm_runtime_get_exception(module_inst);
        fprintf(stderr, "Failed to call get_counters function: %s\n", exception ? exception : "Unknown error");
        wasm_runtime_module_free(inst, accepted_offset);
        wasm_runtime_module_free(inst, rejected_offset);
        return;
    }
    
    // Read the values
    void *native_accepted_ptr = wasm_runtime_addr_app_to_native(inst, accepted_offset);
    void *native_rejected_ptr = wasm_runtime_addr_app_to_native(inst, rejected_offset);
    
    // Copy values to output parameters
    *accepted = *(uint64_t*)native_accepted_ptr;
    *rejected = *(uint64_t*)native_rejected_ptr;
    
    // Free allocated memory
    wasm_runtime_module_free(inst, accepted_offset);
    wasm_runtime_module_free(inst, rejected_offset);
}

// Cleanup function
__attribute__((destructor))
static void cleanup() {
    if (exec_env) {
        wasm_runtime_destroy_exec_env(exec_env);
        exec_env = NULL;
    }
    
    if (module_inst) {
        wasm_runtime_deinstantiate(module_inst);
        module_inst = NULL;
    }
    
    if (module) {
        wasm_runtime_unload(module);
        module = NULL;
    }
    
    wasm_runtime_destroy();
} 