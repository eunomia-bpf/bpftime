#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

/* LuaJIT headers */
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

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

int module_url_filter(const char *url) {
    // For the dynamic loader, we need to return the correct value:
    // - Return 1 for URLs that should be ALLOWED (HTTP 200)
    // - Return 0 for URLs that should be REJECTED (HTTP 403)
    
    // Our internal url_filter returns:
    // - true if URL passes the filter
    // - false if URL fails the filter
    
    bool internal_result = url_filter(url);
    
    fprintf(stderr, "Lua filter for URL %s: internal_result=%d, returning=%d\n", 
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

// LuaJIT runtime state
static lua_State *L = NULL;

// Buffer for data exchange
static char shared_buffer[4096] = {0};
static size_t shared_buffer_size = 0;

// Function names from Lua module
static const char *INITIALIZE_FUNC = "initialize";
static const char *URL_FILTER_FUNC = "url_filter";
static const char *GET_COUNTERS_FUNC = "get_counters";
static const char *SET_BUFFER_FUNC = "set_buffer";
static const char *GET_BUFFER_FUNC = "get_buffer";

// Helper for error handling
#define CHECK_LUA(L, cond, msg) \
    if (!(cond)) { \
        fprintf(stderr, "Lua error: %s - %s\n", msg, lua_tostring((L), -1)); \
        lua_pop((L), 1); \
        return -1; \
    }

// Initialize LuaJIT runtime and load the module
int initialize(const char *prefix) {
    // Get the path to the Lua module from environment variable
    const char *lua_module_path = getenv("LUA_MODULE_PATH");
    if (lua_module_path == NULL) {
        fprintf(stderr, "LUA_MODULE_PATH environment variable is not set\n");
        return -1;
    }
    
    fprintf(stderr, "Loading Lua module from: %s\n", lua_module_path);
    
    // Clean up any previous Lua state
    if (L != NULL) {
        lua_close(L);
        L = NULL;
    }
    
    // Create a new Lua state
    L = luaL_newstate();
    if (L == NULL) {
        fprintf(stderr, "Failed to create Lua state\n");
        return -1;
    }
    
    // Load Lua standard libraries
    luaL_openlibs(L);
    
    // Load the Lua script
    int status = luaL_dofile(L, lua_module_path);
    if (status != 0) {
        fprintf(stderr, "Failed to load Lua module: %s\n", lua_tostring(L, -1));
        lua_close(L);
        L = NULL;
        return -1;
    }
    
    // Call the initialize function with the prefix
    lua_getglobal(L, INITIALIZE_FUNC);
    if (!lua_isfunction(L, -1)) {
        fprintf(stderr, "Cannot find Lua function: %s\n", INITIALIZE_FUNC);
        lua_close(L);
        L = NULL;
        return -1;
    }
    
    // Push the prefix parameter or nil if NULL
    if (prefix != NULL) {
        lua_pushstring(L, prefix);
    } else {
        lua_pushnil(L);
    }
    
    // Call the function (1 argument, 1 result)
    if (lua_pcall(L, 1, 1, 0) != 0) {
        fprintf(stderr, "Failed to call initialize function: %s\n", lua_tostring(L, -1));
        lua_close(L);
        L = NULL;
        return -1;
    }
    
    // Get the result
    if (!lua_isnumber(L, -1)) {
        fprintf(stderr, "Lua initialize function didn't return a number\n");
        lua_close(L);
        L = NULL;
        return -1;
    }
    
    int result = (int)lua_tointeger(L, -1);
    lua_pop(L, 1);  // Pop the result
    
    fprintf(stderr, "Lua URL filter initialized successfully\n");
    return result;
}

// The URL filter function
bool url_filter(const char *url) {
    if (L == NULL) {
        fprintf(stderr, "Lua runtime not initialized\n");
        return false;
    }
    
    // Get the url_filter function
    lua_getglobal(L, URL_FILTER_FUNC);
    if (!lua_isfunction(L, -1)) {
        fprintf(stderr, "Cannot find Lua function: %s\n", URL_FILTER_FUNC);
        lua_pop(L, 1);
        return false;
    }
    
    // Push the URL parameter
    lua_pushstring(L, url);
    
    // Call the function (1 argument, 1 result)
    if (lua_pcall(L, 1, 1, 0) != 0) {
        fprintf(stderr, "Failed to call url_filter function: %s\n", lua_tostring(L, -1));
        lua_pop(L, 1);
        return false;
    }
    
    // Get the result
    if (!lua_isboolean(L, -1)) {
        fprintf(stderr, "Lua url_filter function didn't return a boolean\n");
        lua_pop(L, 1);
        return false;
    }
    
    bool result = lua_toboolean(L, -1);
    lua_pop(L, 1);  // Pop the result
    
    return result;
}

// Set data in the buffer
int set_buffer(const char *data, size_t size) {
    if (L == NULL) {
        fprintf(stderr, "Lua runtime not initialized\n");
        return -1;
    }
    
    // Get the set_buffer function
    lua_getglobal(L, SET_BUFFER_FUNC);
    if (!lua_isfunction(L, -1)) {
        fprintf(stderr, "Cannot find Lua function: %s\n", SET_BUFFER_FUNC);
        lua_pop(L, 1);
        return -1;
    }
    
    // Push the data parameter
    if (data != NULL && size > 0) {
        lua_pushlstring(L, data, size);
    } else {
        lua_pushnil(L);
    }
    
    // Call the function (1 argument, 1 result)
    if (lua_pcall(L, 1, 1, 0) != 0) {
        fprintf(stderr, "Failed to call set_buffer function: %s\n", lua_tostring(L, -1));
        lua_pop(L, 1);
        return -1;
    }
    
    // Get the result
    if (!lua_isnumber(L, -1)) {
        fprintf(stderr, "Lua set_buffer function didn't return a number\n");
        lua_pop(L, 1);
        return -1;
    }
    
    int result = (int)lua_tointeger(L, -1);
    lua_pop(L, 1);  // Pop the result
    
    return result;
}

// Get data from the buffer
int get_buffer(const char **out_data, size_t *out_size) {
    if (L == NULL) {
        fprintf(stderr, "Lua runtime not initialized\n");
        return -1;
    }
    
    // Get the get_buffer function
    lua_getglobal(L, GET_BUFFER_FUNC);
    if (!lua_isfunction(L, -1)) {
        fprintf(stderr, "Cannot find Lua function: %s\n", GET_BUFFER_FUNC);
        lua_pop(L, 1);
        return -1;
    }
    
    // Call the function (0 arguments, 1 result)
    if (lua_pcall(L, 0, 1, 0) != 0) {
        fprintf(stderr, "Failed to call get_buffer function: %s\n", lua_tostring(L, -1));
        lua_pop(L, 1);
        return -1;
    }
    
    // Get the result
    if (!lua_isstring(L, -1)) {
        fprintf(stderr, "Lua get_buffer function didn't return a string\n");
        lua_pop(L, 1);
        return -1;
    }
    
    size_t len;
    const char *data = lua_tolstring(L, -1, &len);
    
    // Copy the data to our buffer
    if (len > sizeof(shared_buffer) - 1) {
        len = sizeof(shared_buffer) - 1;
    }
    
    memcpy(shared_buffer, data, len);
    shared_buffer[len] = '\0';
    shared_buffer_size = len;
    
    lua_pop(L, 1);  // Pop the result
    
    // Set the output parameters
    *out_data = shared_buffer;
    *out_size = shared_buffer_size;
    
    return 0;
}

// Get statistics counters
void get_counters(uint64_t *accepted, uint64_t *rejected) {
    if (L == NULL) {
        fprintf(stderr, "Lua runtime not initialized\n");
        if (accepted) *accepted = 0;
        if (rejected) *rejected = 0;
        return;
    }
    
    // Get the get_counters function
    lua_getglobal(L, GET_COUNTERS_FUNC);
    if (!lua_isfunction(L, -1)) {
        fprintf(stderr, "Cannot find Lua function: %s\n", GET_COUNTERS_FUNC);
        lua_pop(L, 1);
        if (accepted) *accepted = 0;
        if (rejected) *rejected = 0;
        return;
    }
    
    // Call the function (0 arguments, 2 results)
    if (lua_pcall(L, 0, 2, 0) != 0) {
        fprintf(stderr, "Failed to call get_counters function: %s\n", lua_tostring(L, -1));
        lua_pop(L, 1);
        if (accepted) *accepted = 0;
        if (rejected) *rejected = 0;
        return;
    }
    
    // Get the results
    if (!lua_isnumber(L, -2) || !lua_isnumber(L, -1)) {
        fprintf(stderr, "Lua get_counters function didn't return numbers\n");
        lua_pop(L, 2);
        if (accepted) *accepted = 0;
        if (rejected) *rejected = 0;
        return;
    }
    
    if (accepted) *accepted = (uint64_t)lua_tonumber(L, -2);
    if (rejected) *rejected = (uint64_t)lua_tonumber(L, -1);
    
    lua_pop(L, 2);  // Pop the results
}

// Cleanup function
__attribute__((destructor))
static void cleanup() {
    if (L != NULL) {
        lua_close(L);
        L = NULL;
    }
} 