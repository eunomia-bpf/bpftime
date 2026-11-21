#ifndef _UPROBE_MULTI_H
#define _UPROBE_MULTI_H

struct uprobe_multi_event {
	int is_ret;
	union {
		struct {
			long arg1;
			long arg2;
		} uprobe;
		struct {
			long ret_val;
		} uretprobe;
	};
};

#endif
