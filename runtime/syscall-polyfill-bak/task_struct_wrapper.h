#ifndef _TESK_STRUCT_WRAPPER_H
#define _TESK_STRUCT_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

struct TaskStructWrapper {
	struct task_struct *curr;
	struct task_struct *parent;
};
struct TaskStructWrapper *task_struct_wrapper_create();
void task_struct_wrapper_destroy(struct TaskStructWrapper *st);
#ifdef __cplusplus
}
#endif

#endif
