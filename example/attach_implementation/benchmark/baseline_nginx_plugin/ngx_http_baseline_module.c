#include "ngx_conf_file.h"
#include "ngx_log.h"
#include "ngx_string.h"
#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#include <stdio.h>
#include <string.h>

static char bad_url_prefix[128] = "/";

// Simple string prefix check function
static int str_startswith(const char* main, const char* prefix) 
{
    size_t prefix_len = strlen(prefix);
    size_t main_len = strlen(main);
    
    if (main_len < prefix_len) {
        return 0;
    }
    
    return strncmp(main, prefix, prefix_len) == 0 ? 1 : 0;
}

int baseline_url_filter(const char* url)
{
    // Return 1 if the URL starts with the bad prefix (allow), 0 otherwise (deny)
    return str_startswith(url, bad_url_prefix);
}

int baseline_initialize(const char* prefix)
{
    if (prefix != NULL) {
        strncpy(bad_url_prefix, prefix, sizeof(bad_url_prefix) - 1);
        bad_url_prefix[sizeof(bad_url_prefix) - 1] = '\0';
    }
    
    printf("Baseline URL filter initialized with prefix: %s\n", bad_url_prefix);
    return 0;
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
    NULL,                          /* exit master */
    NGX_MODULE_V1_PADDING
};

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
    ngx_http_baseline_loc_conf_t *blcf;

    cmcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_core_module);

    h = ngx_array_push(&cmcf->phases[NGX_HTTP_ACCESS_PHASE].handlers);
    if (h == NULL) {
        return NGX_ERROR;
    }

    *h = ngx_http_baseline_handler;
    
    // Get the location configuration to access the prefix
    blcf = ngx_http_conf_get_module_loc_conf(cf, ngx_http_baseline_module);
    
    // Initialize the baseline filter with the configured prefix
    char prefix_buf[128] = "/";  // Default prefix
    if (blcf->prefix.data != NULL && blcf->prefix.len > 0) {
        ngx_snprintf((u_char*)prefix_buf, sizeof(prefix_buf) - 1, "%V", &blcf->prefix);
        prefix_buf[blcf->prefix.len] = '\0';
    }
    
    int err = baseline_initialize(prefix_buf);
    ngx_log_error(NGX_LOG_INFO, cf->log, 0, "Baseline module initialized with prefix '%s': %d", 
                  prefix_buf, err);
    
    return NGX_OK;
}
