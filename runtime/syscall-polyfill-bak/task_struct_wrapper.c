#include "vmlinux_x86-64.h"

#include "task_struct_wrapper.h"

typedef unsigned long size_t;

void *malloc(size_t size);
void *memset(void *s, int c, size_t n);
void free(void *ptr);
pid_t getpid(void);

struct TaskStructWrapper *task_struct_wrapper_create()
{
	struct TaskStructWrapper *wrapper = (struct TaskStructWrapper *)malloc(
		sizeof(struct TaskStructWrapper));

	wrapper->curr = malloc(sizeof(struct task_struct));
	memset(wrapper->curr, 0, sizeof(struct task_struct));
	wrapper->parent = malloc(sizeof(struct task_struct));
	memset(wrapper->parent, 0, sizeof(struct task_struct));
	wrapper->curr->real_parent = wrapper->parent;
	wrapper->parent->tgid = getpid();
	return wrapper;
}

void task_struct_wrapper_destroy(struct TaskStructWrapper *st)
{
	free(st->curr);
	free(st->parent);
	free(st);
}
