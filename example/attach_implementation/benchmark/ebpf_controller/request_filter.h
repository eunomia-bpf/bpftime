#ifndef _REQUEST_FILTER_H
#define _REQUEST_FILTER_H

struct request_filter_argument {
	const char *url_to_check;
	const char *accept_prefix;
};

struct request_counter {
	__u64 accepted_count;
	__u64 rejected_count;
};

#endif
