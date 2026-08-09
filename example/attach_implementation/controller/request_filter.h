#ifndef _REQUEST_FILTER_H
#define _REQUEST_FILTER_H

struct request_filter_event {
	char url[128];
	int accepted;
};

struct request_filter_argument {
	const char *url_to_check;
	const char *accept_prefix;
};

#endif
