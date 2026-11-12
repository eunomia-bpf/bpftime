#include "ngx_conf_file.h"
#include "ngx_log.h"
#include "ngx_string.h"
#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>

// Function pointers for dynamically loaded functions
typedef int (*url_filter_func_t)(const char *url);
typedef int (*initialize_func_t)(const char *prefix);
typedef void (*get_counters_func_t)(uint64_t *accepted, uint64_t *rejected);

// Module context to store loaded library and function pointers
typedef struct {
    void *lib_handle;
    url_filter_func_t url_filter;
    initialize_func_t initialize;
    get_counters_func_t get_counters;
    uint64_t accepted_count;
    uint64_t rejected_count;
} dynamic_module_ctx_t;

static dynamic_module_ctx_t *module_ctx = NULL;

// Path to the filter library
static const char *filter_lib_path = NULL;

typedef struct {
    ngx_flag_t enable;
    ngx_str_t  prefix;
    ngx_str_t  lib_path;
} ngx_http_dynamic_load_loc_conf_t;

static ngx_int_t ngx_http_dynamic_load_handler(ngx_http_request_t *r);
static void *ngx_http_dynamic_load_create_loc_conf(ngx_conf_t *cf);
static char *ngx_http_dynamic_load_merge_loc_conf(ngx_conf_t *cf, void *parent, void *child);
static ngx_int_t ngx_http_dynamic_load_init(ngx_conf_t *cf);
static char *ngx_http_dynamic_load_set_prefix(ngx_conf_t *cf, ngx_command_t *cmd, void *conf);
static char *ngx_http_dynamic_load_set_lib_path(ngx_conf_t *cf, ngx_command_t *cmd, void *conf);
static void ngx_http_dynamic_load_exit(ngx_cycle_t *cycle);

static ngx_command_t ngx_http_dynamic_load_commands[] = {
    { ngx_string("dynamic_load_request_filter"),
      NGX_HTTP_MAIN_CONF | NGX_HTTP_SRV_CONF | NGX_HTTP_LOC_CONF | NGX_CONF_FLAG,
      ngx_conf_set_flag_slot,
      NGX_HTTP_LOC_CONF_OFFSET,
      offsetof(ngx_http_dynamic_load_loc_conf_t, enable),
      NULL },

    { ngx_string("dynamic_load_url_prefix"),
      NGX_HTTP_MAIN_CONF | NGX_HTTP_SRV_CONF | NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
      ngx_http_dynamic_load_set_prefix,
      NGX_HTTP_LOC_CONF_OFFSET,
      offsetof(ngx_http_dynamic_load_loc_conf_t, prefix),
      NULL },
      
    { ngx_string("dynamic_load_lib_path"),
      NGX_HTTP_MAIN_CONF | NGX_HTTP_SRV_CONF | NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
      ngx_http_dynamic_load_set_lib_path,
      NGX_HTTP_LOC_CONF_OFFSET,
      offsetof(ngx_http_dynamic_load_loc_conf_t, lib_path),
      NULL },

    ngx_null_command
};

static ngx_http_module_t ngx_http_dynamic_load_module_ctx = {
    NULL,                           /* preconfiguration */
    ngx_http_dynamic_load_init,     /* postconfiguration */

    NULL,                           /* create main configuration */
    NULL,                           /* init main configuration */

    NULL,                           /* create server configuration */
    NULL,                           /* merge server configuration */

    ngx_http_dynamic_load_create_loc_conf,  /* create location configuration */
    ngx_http_dynamic_load_merge_loc_conf    /* merge location configuration */
};

ngx_module_t ngx_http_dynamic_load_module = {
    NGX_MODULE_V1,
    &ngx_http_dynamic_load_module_ctx,  /* module context */
    ngx_http_dynamic_load_commands,     /* module directives */
    NGX_HTTP_MODULE,                    /* module type */
    NULL,                               /* init master */
    NULL,                               /* init module */
    NULL,                               /* init process */
    NULL,                               /* init thread */
    NULL,                               /* exit thread */
    NULL,                               /* exit process */
    ngx_http_dynamic_load_exit,         /* exit master */
    NGX_MODULE_V1_PADDING
};

// Initialize dynamic module context and load the filter library
static int init_dynamic_module(const char *lib_path, const char *prefix)
{
    if (module_ctx) {
        // Module already initialized
        return 0;
    }
    
    module_ctx = (dynamic_module_ctx_t *)malloc(sizeof(dynamic_module_ctx_t));
    if (!module_ctx) {
        fprintf(stderr, "Failed to allocate module context\n");
        return -1;
    }
    
    memset(module_ctx, 0, sizeof(dynamic_module_ctx_t));
    
    // Load the library
    module_ctx->lib_handle = dlopen(lib_path, RTLD_LAZY);
    if (!module_ctx->lib_handle) {
        fprintf(stderr, "Failed to load dynamic library: %s\n", dlerror());
        free(module_ctx);
        module_ctx = NULL;
        return -1;
    }
    
    // Load the functions
    module_ctx->url_filter = (url_filter_func_t)dlsym(module_ctx->lib_handle, "module_url_filter");
    if (!module_ctx->url_filter) {
        fprintf(stderr, "Failed to load module_url_filter function: %s\n", dlerror());
        dlclose(module_ctx->lib_handle);
        free(module_ctx);
        module_ctx = NULL;
        return -1;
    }
    
    module_ctx->initialize = (initialize_func_t)dlsym(module_ctx->lib_handle, "module_initialize");
    if (!module_ctx->initialize) {
        fprintf(stderr, "Failed to load module_initialize function: %s\n", dlerror());
        dlclose(module_ctx->lib_handle);
        free(module_ctx);
        module_ctx = NULL;
        return -1;
    }
    
    module_ctx->get_counters = (get_counters_func_t)dlsym(module_ctx->lib_handle, "module_get_counters");
    if (!module_ctx->get_counters) {
        fprintf(stderr, "Failed to load module_get_counters function: %s\n", dlerror());
        dlclose(module_ctx->lib_handle);
        free(module_ctx);
        module_ctx = NULL;
        return -1;
    }
    
    // Initialize the module
    int ret = module_ctx->initialize(prefix);
    if (ret != 0) {
        fprintf(stderr, "Failed to initialize the module: %d\n", ret);
        dlclose(module_ctx->lib_handle);
        free(module_ctx);
        module_ctx = NULL;
        return -1;
    }
    
    return 0;
}

// Cleanup the dynamic module context
static void cleanup_dynamic_module()
{
    if (module_ctx) {
        if (module_ctx->lib_handle) {
            dlclose(module_ctx->lib_handle);
        }
        free(module_ctx);
        module_ctx = NULL;
    }
}

static void ngx_http_dynamic_load_exit(ngx_cycle_t *cycle)
{
    cleanup_dynamic_module();
}

static char *ngx_http_dynamic_load_set_prefix(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    ngx_http_dynamic_load_loc_conf_t *dlcf = conf;
    ngx_str_t *value;

    value = cf->args->elts;
    dlcf->prefix = value[1];

    return NGX_CONF_OK;
}

static char *ngx_http_dynamic_load_set_lib_path(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    ngx_http_dynamic_load_loc_conf_t *dlcf = conf;
    ngx_str_t *value;

    value = cf->args->elts;
    dlcf->lib_path = value[1];
    
    // Store the library path for later use
    if (filter_lib_path == NULL) {
        filter_lib_path = strndup((const char *)dlcf->lib_path.data, dlcf->lib_path.len);
    }

    return NGX_CONF_OK;
}

static ngx_int_t ngx_http_dynamic_load_handler(ngx_http_request_t *r)
{
    ngx_http_dynamic_load_loc_conf_t *dlcf;

    dlcf = ngx_http_get_module_loc_conf(r, ngx_http_dynamic_load_module);
    
    if (dlcf->enable && module_ctx && module_ctx->url_filter) {
        char buf[512];
        snprintf(buf, sizeof(buf), "%*s", (int)r->uri.len, r->uri.data);

        // Clean up the URL string
        int len = strlen(buf);
        for (int i = len - 1; i >= 0; i--) {
            if (buf[i] == ' ') {
                buf[i] = '\0';
                break;
            }
        }
        
        // Call the filter function from the loaded library
        int ret = module_ctx->url_filter(buf);

        // Update local counters for easier access
        if (module_ctx->get_counters) {
            module_ctx->get_counters(&module_ctx->accepted_count, &module_ctx->rejected_count);
        }

        // Same return values as the other implementations
        if (ret == 0) {
            return NGX_HTTP_FORBIDDEN;
        }
    }

    return NGX_DECLINED;
}

static void *ngx_http_dynamic_load_create_loc_conf(ngx_conf_t *cf)
{
    ngx_http_dynamic_load_loc_conf_t *conf;

    conf = ngx_pcalloc(cf->pool, sizeof(ngx_http_dynamic_load_loc_conf_t));
    if (conf == NULL) {
        return NULL;
    }

    conf->enable = NGX_CONF_UNSET;
    conf->prefix.data = NULL;
    conf->prefix.len = 0;
    conf->lib_path.data = NULL;
    conf->lib_path.len = 0;

    return conf;
}

static char *ngx_http_dynamic_load_merge_loc_conf(ngx_conf_t *cf, void *parent, void *child)
{
    ngx_http_dynamic_load_loc_conf_t *prev = parent;
    ngx_http_dynamic_load_loc_conf_t *conf = child;

    ngx_conf_merge_value(conf->enable, prev->enable, 0);
    
    if (conf->prefix.data == NULL) {
        conf->prefix = prev->prefix;
    }
    
    if (conf->lib_path.data == NULL) {
        conf->lib_path = prev->lib_path;
    }

    return NGX_CONF_OK;
}

static ngx_int_t ngx_http_dynamic_load_init(ngx_conf_t *cf)
{
    ngx_http_handler_pt *h;
    ngx_http_core_main_conf_t *cmcf;
    ngx_http_dynamic_load_loc_conf_t *dlcf;

    // Print environment variables for debugging
    fprintf(stderr, "DEBUG: Checking environment variables:\n");
    fprintf(stderr, "DEBUG: DYNAMIC_LOAD_LIB_PATH=%s\n", getenv("DYNAMIC_LOAD_LIB_PATH") ? getenv("DYNAMIC_LOAD_LIB_PATH") : "not set");
    fprintf(stderr, "DEBUG: DYNAMIC_LOAD_URL_PREFIX=%s\n", getenv("DYNAMIC_LOAD_URL_PREFIX") ? getenv("DYNAMIC_LOAD_URL_PREFIX") : "not set");

    cmcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_core_module);

    h = ngx_array_push(&cmcf->phases[NGX_HTTP_ACCESS_PHASE].handlers);
    if (h == NULL) {
        return NGX_ERROR;
    }

    *h = ngx_http_dynamic_load_handler;
    
    // Get the location configuration to access the prefix and lib path
    dlcf = ngx_http_conf_get_module_loc_conf(cf, ngx_http_dynamic_load_module);
    
    // Initialize the dynamic module and load the filter library
    char prefix_buf[128] = "/";  // Default prefix
    
    // Check environment variable for URL prefix first
    const char *env_prefix = getenv("DYNAMIC_LOAD_URL_PREFIX");
    if (env_prefix != NULL) {
        strncpy(prefix_buf, env_prefix, sizeof(prefix_buf) - 1);
        prefix_buf[sizeof(prefix_buf) - 1] = '\0';
        ngx_log_error(NGX_LOG_INFO, cf->log, 0, 
                    "Using URL prefix from environment: %s", prefix_buf);
    }
    // If not set via environment, use the config if available
    else if (dlcf->prefix.data != NULL && dlcf->prefix.len > 0) {
        ngx_snprintf((u_char*)prefix_buf, sizeof(prefix_buf) - 1, "%V", &dlcf->prefix);
        prefix_buf[dlcf->prefix.len] = '\0';
    }
    
    // Check environment variable for library path first
    const char *env_lib_path = getenv("DYNAMIC_LOAD_LIB_PATH");
    char lib_path_buf[512] = {0};
    
    if (env_lib_path != NULL) {
        strncpy(lib_path_buf, env_lib_path, sizeof(lib_path_buf) - 1);
        lib_path_buf[sizeof(lib_path_buf) - 1] = '\0';
        ngx_log_error(NGX_LOG_INFO, cf->log, 0, 
                    "Using library path from environment: %s", lib_path_buf);
    }
    // If not set via environment, use the config if available
    else if (dlcf->lib_path.data != NULL && dlcf->lib_path.len > 0) {
        ngx_snprintf((u_char*)lib_path_buf, sizeof(lib_path_buf) - 1, "%V", &dlcf->lib_path);
        lib_path_buf[dlcf->lib_path.len] = '\0';
    }
    else {
        ngx_log_error(NGX_LOG_ERR, cf->log, 0, 
                    "No library path specified for dynamic_load module (set DYNAMIC_LOAD_LIB_PATH environment variable)");
        return NGX_ERROR;
    }
    
    // More debugging output
    fprintf(stderr, "DEBUG: Using prefix: %s\n", prefix_buf);
    fprintf(stderr, "DEBUG: Using lib_path: %s\n", lib_path_buf);

    int err = init_dynamic_module(lib_path_buf, prefix_buf);
    
    if (module_ctx != NULL) {
        ngx_log_error(NGX_LOG_INFO, cf->log, 0, 
                    "Dynamic load module initialized with prefix '%s': %d (accepted:%lu, rejected:%lu)", 
                    prefix_buf, err, 
                    module_ctx->accepted_count, module_ctx->rejected_count);
    } else {
        ngx_log_error(NGX_LOG_ERR, cf->log, 0, 
                    "Dynamic load module failed to initialize library");
    }
    
    return NGX_OK;
} 