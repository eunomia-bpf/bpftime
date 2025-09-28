#ifndef _get_stack_id_example_H
#define _get_stack_id_example_H

enum event_operation {
	MALLOC_ENTER,
	FREE_ENTER,

};

struct event {
	int pid;
	int stack_id;
	void *addr;
	enum event_operation operation;
};
#endif
