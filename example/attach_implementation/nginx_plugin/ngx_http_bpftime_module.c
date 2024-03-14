#include "ngx_conf_file.h"
#include "ngx_log.h"
#include "ngx_string.h"
#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>

int nginx_plugin_example_run_filter(const char *url);
int nginx_plugin_example_initialize();
typedef struct {
	ngx_flag_t enable;
} ngx_http_bpftime_loc_conf_t;

static ngx_int_t ngx_http_bpftime_handler(ngx_http_request_t *r);
static void *ngx_http_bpftime_create_loc_conf(ngx_conf_t *cf);
static char *ngx_http_bpftime_merge_loc_conf(ngx_conf_t *cf, void *parent,
					     void *child);
static ngx_int_t ngx_http_bpftime_init(ngx_conf_t *cf);

static ngx_command_t ngx_http_bpftime_commands[] = {

	{ ngx_string("bpftime_request_filter"),
	  NGX_HTTP_MAIN_CONF | NGX_HTTP_SRV_CONF | NGX_HTTP_LOC_CONF |
		  NGX_CONF_FLAG,
	  ngx_conf_set_flag_slot, NGX_HTTP_LOC_CONF_OFFSET,
	  offsetof(ngx_http_bpftime_loc_conf_t, enable), NULL },

	ngx_null_command
};

static ngx_http_module_t ngx_http_bpftime_module_ctx = {
	NULL, /* preconfiguration */
	ngx_http_bpftime_init, /* postconfiguration */

	NULL, /* create main configuration */
	NULL, /* init main configuration */

	NULL, /* create server configuration */
	NULL, /* merge server configuration */

	ngx_http_bpftime_create_loc_conf, /* create location configuration */
	ngx_http_bpftime_merge_loc_conf /* merge location configuration */
};

ngx_module_t ngx_http_bpftime_module = {
	NGX_MODULE_V1,
	&ngx_http_bpftime_module_ctx, /* module
					 context
				       */
	ngx_http_bpftime_commands, /* module
				      directives
				    */
	NGX_HTTP_MODULE, /* module type */
	NULL, /* init master */
	NULL, /* init module */
	NULL, /* init process */
	NULL, /* init thread */
	NULL, /* exit thread */
	NULL, /* exit process */
	NULL, /* exit master */
	NGX_MODULE_V1_PADDING
};

static ngx_int_t ngx_http_bpftime_handler(ngx_http_request_t *r)
{
	ngx_http_bpftime_loc_conf_t *ulcf;

	ngx_log_error(NGX_LOG_DEBUG, r->connection->log, 0,
		      "http ua access handler");

	ulcf = ngx_http_get_module_loc_conf(r, ngx_http_bpftime_module);
	if (ulcf->enable) {
		char buf[512];
		snprintf(buf, sizeof(buf), "%*s", (int)r->uri.len, r->uri.data);

		int len = strlen(buf);
		for (int i = len - 1; i >= 0; i--)
			if (buf[i] == ' ') {
				buf[i] = '\0';
				break;
			}
		len = strlen(buf);
		ngx_log_error(NGX_LOG_DEBUG, r->connection->log, 0,
			      "Accessed uri: %s", buf);

		int ret = nginx_plugin_example_run_filter(buf);

		ngx_log_error(NGX_LOG_DEBUG, r->connection->log, 0,
			      "Ebpf ret: %d", (int)ret);
		if (ret == 0) {
			ngx_log_error(NGX_LOG_DEBUG, r->connection->log, 0,
				      "Rejected access by ebpf");
			return NGX_HTTP_FORBIDDEN;
		}
	}

	return NGX_DECLINED;
}

static void *ngx_http_bpftime_create_loc_conf(ngx_conf_t *cf)
{
	ngx_http_bpftime_loc_conf_t *conf;

	conf = ngx_pcalloc(cf->pool, sizeof(ngx_http_bpftime_loc_conf_t));
	if (conf == NULL) {
		return NULL;
	}

	conf->enable = NGX_CONF_UNSET;

	return conf;
}

static char *ngx_http_bpftime_merge_loc_conf(ngx_conf_t *cf, void *parent,
					     void *child)
{
	ngx_http_bpftime_loc_conf_t *prev = parent;
	ngx_http_bpftime_loc_conf_t *conf = child;

	ngx_conf_merge_value(conf->enable, prev->enable, 0);

	return NGX_CONF_OK;
}

static ngx_int_t ngx_http_bpftime_init(ngx_conf_t *cf)
{
	ngx_http_handler_pt *h;
	ngx_http_core_main_conf_t *cmcf;

	cmcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_core_module);

	h = ngx_array_push(&cmcf->phases[NGX_HTTP_ACCESS_PHASE].handlers);
	if (h == NULL) {
		return NGX_ERROR;
	}

	*h = ngx_http_bpftime_handler;
	int err = nginx_plugin_example_initialize();
	ngx_log_error(NGX_LOG_ERR, cf->log, 0, "Module init: %d", err);
	return NGX_OK;
}

void dummy()
{
}
