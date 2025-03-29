
#include "log.h"
#include "defs.h"
#include "command.h"
#include "cli/cli-cmds.h"
#include "value.h"

#include "cricket-heap.h"

bool cricket_focus_host(bool batch_flag)
{
    struct cmd_list_element *c;
    const char *threadcmd = "thread 1";
    c = lookup_cmd(&threadcmd, cmdlist, "", 0, 1);

    if (c == NULL || c == (struct cmd_list_element *)-1)
        return false;

    if (!cmd_func_p(c))
        return false;

    cmd_func(c, threadcmd, !batch_flag);
    return true;
}

bool cricket_focus_kernel(bool batch_flag)
{
    struct cmd_list_element *c;
    const char *threadcmd = "cuda kernel 0";
    c = lookup_cmd(&threadcmd, cmdlist, "", 0, 1);

    if (c == NULL || c == (struct cmd_list_element *)-1)
        return false;

    if (!cmd_func_p(c))
        return false;

    cmd_func(c, threadcmd, !batch_flag);
    return true;
}

bool cricket_heap_memreg_size(void *addr, size_t *size)
{
    char *callstr = NULL;
    struct value *val;
    struct type *type;

    if (size == NULL)
        return false;

    if (asprintf(&callstr, "(size_t)api_records_malloc_get_size(0x%llx)", addr) == -1)
        return false;

    if (callstr == NULL) {
        LOGE(LOG_ERROR, "asprintf returned NULL");
        return false;
    }
    LOGE(LOG_DEBUG, "callstr: %s", callstr);
    *size = parse_and_eval_long(callstr);
    free(callstr);

    if (*size == 0)
        return false;

    return true;
}
