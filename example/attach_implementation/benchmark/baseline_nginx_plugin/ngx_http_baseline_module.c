#include "ngx_conf_file.h"
#include "ngx_log.h"
#include "ngx_string.h"
#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

// Shared memory structure - must match the one in baseline_controller.cpp
typedef struct {
    char accept_url_prefix[128];
    uint64_t accepted_count;
    uint64_t rejected_count;
} baseline_shared_data_t;

static const char* SHM_NAME = "/baseline_nginx_filter_shm";
static baseline_shared_data_t* shared_data = NULL;
static int shm_fd = -1;

// String prefix check function - exact same implementation as in the eBPF code
static int str_startswith(const char *main, const char *pat)
{
    int i = 0;
    while (*main == *pat && *main != 0 && *pat != 0 && i++ < 128) {
        main++;
        pat++;
    }
    return *pat == 0;
}

// Initialize shared memory - open existing shared memory created by controller
static int init_shared_memory() 
{
    // Open existing shared memory (should be created by controller)
    shm_fd = shm_open(SHM_NAME, O_RDWR, 0666);
    if (shm_fd == -1) {
        fprintf(stderr, "Failed to open shared memory: %s\n", strerror(errno));
        fprintf(stderr, "Make sure the controller is running before starting Nginx\n");
        return -1;
    }

    // Map shared memory segment into our address space
    shared_data = (baseline_shared_data_t*)mmap(
        NULL,
        sizeof(baseline_shared_data_t),
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        shm_fd,
        0
    );

    if (shared_data == MAP_FAILED) {
        fprintf(stderr, "Failed to map shared memory: %s\n", strerror(errno));
        close(shm_fd);
        return -1;
    }

    // We don't need to initialize values; they are already set by the controller
    return 0;
}

// Clean up shared memory
static void cleanup_shared_memory() 
{
    if (shared_data != NULL && shared_data != MAP_FAILED) {
        munmap(shared_data, sizeof(baseline_shared_data_t));
        shared_data = NULL;
    }
    
    if (shm_fd != -1) {
        close(shm_fd);
        shm_fd = -1;
    }
    
    // Do NOT unlink the shared memory, as the controller is responsible for that
}

int baseline_url_filter(const char *url)
{
    if (shared_data == NULL) {
        return 1; // Default to accepting if shared memory not initialized
    }

    // Use the same logic as the eBPF filter
    int result = str_startswith(url, shared_data->accept_url_prefix);
    
    // Update counters (same as eBPF implementation)
    if (result) {
        __sync_fetch_and_add(&shared_data->accepted_count, 1); // Thread-safe increment
    } else {
        __sync_fetch_and_add(&shared_data->rejected_count, 1); // Thread-safe increment
    }
    
    return result;
}

int baseline_initialize(const char *prefix)
{
    // Initialize shared memory - opens existing shared memory created by controller
    if (init_shared_memory() != 0) {
        fprintf(stderr, "Failed to initialize shared memory\n");
        return -1;
    }
    
    // No need to set prefix, just read from shared memory what controller set
    printf("Baseline URL filter initialized with prefix from controller: %s\n", 
           shared_data->accept_url_prefix);
    return 0;
} 

// Function to get the current counter values (used for benchmarking)
void baseline_get_counters(uint64_t *accepted, uint64_t *rejected)
{
    if (shared_data == NULL) {
        if (accepted) *accepted = 0;
        if (rejected) *rejected = 0;
        return;
    }

    if (accepted) *accepted = shared_data->accepted_count;
    if (rejected) *rejected = shared_data->rejected_count;
}

typedef struct {
    ngx_flag_t enable;
    ngx_str_t  prefix;
} ngx_http_baseline_loc_conf_t;

static ngx_int_t ngx_http_baseline_handler(ngx_http_request_t *r);
static void *ngx_http_baseline_create_loc_conf(ngx_conf_t *cf);
static char *ngx_http_baseline_merge_loc_conf(ngx_conf_t *cf, void *parent, void *child);
static ngx_int_t ngx_http_baseline_init(ngx_conf_t *cf);
static char *ngx_http_baseline_set_prefix(ngx_conf_t *cf, ngx_command_t *cmd, void *conf);
static void ngx_http_baseline_exit(ngx_cycle_t *cycle);

static ngx_command_t ngx_http_baseline_commands[] = {
    { ngx_string("baseline_request_filter"),
      NGX_HTTP_MAIN_CONF | NGX_HTTP_SRV_CONF | NGX_HTTP_LOC_CONF | NGX_CONF_FLAG,
      ngx_conf_set_flag_slot,
      NGX_HTTP_LOC_CONF_OFFSET,
      offsetof(ngx_http_baseline_loc_conf_t, enable),
      NULL },

    { ngx_string("baseline_url_prefix"),
      NGX_HTTP_MAIN_CONF | NGX_HTTP_SRV_CONF | NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
      ngx_http_baseline_set_prefix,
      NGX_HTTP_LOC_CONF_OFFSET,
      offsetof(ngx_http_baseline_loc_conf_t, prefix),
      NULL },

    ngx_null_command
};

static ngx_http_module_t ngx_http_baseline_module_ctx = {
    NULL,                          /* preconfiguration */
    ngx_http_baseline_init,        /* postconfiguration */

    NULL,                          /* create main configuration */
    NULL,                          /* init main configuration */

    NULL,                          /* create server configuration */
    NULL,                          /* merge server configuration */

    ngx_http_baseline_create_loc_conf, /* create location configuration */
    ngx_http_baseline_merge_loc_conf   /* merge location configuration */
};

ngx_module_t ngx_http_baseline_module = {
    NGX_MODULE_V1,
    &ngx_http_baseline_module_ctx, /* module context */
    ngx_http_baseline_commands,    /* module directives */
    NGX_HTTP_MODULE,               /* module type */
    NULL,                          /* init master */
    NULL,                          /* init module */
    NULL,                          /* init process */
    NULL,                          /* init thread */
    NULL,                          /* exit thread */
    NULL,                          /* exit process */
    ngx_http_baseline_exit,        /* exit master */
    NGX_MODULE_V1_PADDING
};

static void ngx_http_baseline_exit(ngx_cycle_t *cycle)
{
    cleanup_shared_memory();
}

static char *ngx_http_baseline_set_prefix(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    ngx_http_baseline_loc_conf_t *blcf = conf;
    ngx_str_t *value;

    value = cf->args->elts;
    blcf->prefix = value[1];

    return NGX_CONF_OK;
}

static ngx_int_t ngx_http_baseline_handler(ngx_http_request_t *r)
{
    ngx_http_baseline_loc_conf_t *blcf;

    blcf = ngx_http_get_module_loc_conf(r, ngx_http_baseline_module);
    
    if (blcf->enable) {
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
        
        // Call the filter function
        int ret = baseline_url_filter(buf);

        // Same return values as the eBPF implementation
        if (ret == 0) {
            return NGX_HTTP_FORBIDDEN;
        }
    }

    return NGX_DECLINED;
}

static void *ngx_http_baseline_create_loc_conf(ngx_conf_t *cf)
{
    ngx_http_baseline_loc_conf_t *conf;

    conf = ngx_pcalloc(cf->pool, sizeof(ngx_http_baseline_loc_conf_t));
    if (conf == NULL) {
        return NULL;
    }

    conf->enable = NGX_CONF_UNSET;
    conf->prefix.data = NULL;
    conf->prefix.len = 0;

    return conf;
}

static char *ngx_http_baseline_merge_loc_conf(ngx_conf_t *cf, void *parent, void *child)
{
    ngx_http_baseline_loc_conf_t *prev = parent;
    ngx_http_baseline_loc_conf_t *conf = child;

    ngx_conf_merge_value(conf->enable, prev->enable, 0);
    
    if (conf->prefix.data == NULL) {
        conf->prefix = prev->prefix;
    }

    return NGX_CONF_OK;
}

static ngx_int_t ngx_http_baseline_init(ngx_conf_t *cf)
{
    ngx_http_handler_pt *h;
    ngx_http_core_main_conf_t *cmcf;

    cmcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_core_module);

    h = ngx_array_push(&cmcf->phases[NGX_HTTP_ACCESS_PHASE].handlers);
    if (h == NULL) {
        return NGX_ERROR;
    }

    *h = ngx_http_baseline_handler;
    
    // Initialize the baseline filter - connect to shared memory created by controller
    int err = baseline_initialize(NULL);
    
    if (shared_data != NULL) {
        ngx_log_error(NGX_LOG_INFO, cf->log, 0, 
                    "Baseline module initialized using controller's shared memory. Prefix: '%s' (accepted:%lu, rejected:%lu)", 
                    shared_data->accept_url_prefix, err, 
                    shared_data->accepted_count, shared_data->rejected_count);
    } else {
        ngx_log_error(NGX_LOG_ERR, cf->log, 0, 
                    "Baseline module failed to connect to shared memory. Make sure controller is running.");
    }
    
    return NGX_OK;
}
