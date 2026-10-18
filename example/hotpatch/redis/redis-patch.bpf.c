#define BPF_NO_GLOBAL_DATA
#include "vmlinux.h"
#include "redis-server.h"
#include "bpf/bpf_tracing.h"
#include "bpf/bpf_helpers.h"
#include "ufunc.bpf.h"

#define OBJ_STREAM 6 /* Stream object. */

int checkType(client *c, robj *o, int type);

robj *lookupKeyWriteOrReply(client *c, robj *key, robj *reply);

#ifdef HOST_IS_32
int read_argc(void *base)
{
	return *(int *)(((char *)base) + 0x20);
}
void *read_argv(void *base, int idx)
{
	unsigned ret = *(unsigned *)(((char *)base) + 0x24 + idx * 4);
	return (void *)ret;
}
void *read_db(void *base)
{
	unsigned ret = *(unsigned *)(((char *)base) + 0xc);
	return (void *)ret;
}
unsigned int read_type(void *base)
{
	return *(unsigned int *)(((char *)base) + 0x00);
}
#else
static inline int read_argc(void *base)
{
	return ((client *)base)->argc;
}
static inline void *read_argv(void *base, int idx)
{
	return ((client *)base)->argv[idx];
}
static inline void *read_db(void *base)
{
	return ((client *)base)->db;
}
static inline unsigned int read_type(void *base)
{
	// It's a bit field..
	unsigned a = ((robj *)base)->type;
	return a & 0x0f;
}
#endif

SEC("uprobe/redis-server:xgroupCommand")
int BPF_UPROBE(xgroupCommand, client *c)
{
	bpf_printk("xgroupCommand: %p\n", c);
	// char *opt = c->argv[1]->ptr; /* Subcommand name. */
	/* Lookup the key now, this is common for all the subcommands but HELP.
	 */
	int argc = read_argc(c);
	if (argc >= 4) {
		bpf_printk("c->argc >= 4\n");
		void *argv2 = read_argv(c, 2);
		robj *o = lookupKeyWriteOrReply(c, argv2, NULL);
		if (o == NULL || checkType(c, o, OBJ_STREAM)) {
			bpf_printk("SKIP\n");
			bpf_override_return(NULL, 0);
			return 0;
		}
	}
	bpf_printk("RESUME\n");
	return 0;
}

robj *lookupKeyWriteOrReply(client *c, robj *key, robj *reply)
{
	bpf_printk("lookupKeyWriteOrReply: %p %p %p\n", c, key, reply);
	void *db = read_db(c);
	robj *o = (robj *)UFUNC_CALL_NAME_2("lookupKeyWrite", db, key);
	bpf_printk("lookupKeyWriteOrReply ufunc return: %p\n", o);
	return o;
}

int checkType(client *c, robj *o, int type)
{
	unsigned ty = read_type(o);
	if (ty != type) {
		return 1;
	}
	return 0;
}
