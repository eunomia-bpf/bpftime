# redis example

The pre-build binary is for x86 in tar.gz.

you can compile one in `redis-5.0-rc1`.

1. find function

     ```console
     $ nm ./redis-server | grep xgroupCommand
     00000000000b7c80 T xgroupCommand
     00000000000280b9 t xgroupCommand.cold
     ```

2. generate btf (optional)

     ```sh
     pahole --btf_encode_detached redis-server.btf ./redis-server
     ```

3. generate header

     ```sh
     bpftool btf dump file redis-server.btf format c > redis-server.btf.h
     ```

4. trigger the POC(before the patch):

     PoC:

     ```console
     $ ./redis-server

     # on another bash
     $ ./redis-cli -p 6379
     127.0.0.1:1234> set a 123
     OK
     127.0.0.1:1234> xgroup create a b $
     Error: Connection reset by peer  <— segfault'ed
     127.0.0.1:1234>

     The bug also could be triggered via netcat
     $ nc 127.0.0.1 1234
     set a 123
     +OK
     xgroup create a b $  <— segfault’ed after this line
     ```

5. prepare patch:

     This is fixed by commit:

     ```c
     @@ -1576,7 +1576,7 @@ NULL
          /* Lookup the key now, this is common for all the subcommands but HELP. */
          if (c->argc >= 4) {
     robj *o = lookupKeyWriteOrReply(c,c->argv[2],shared.nokeyerr);
     -         if (o == NULL) return;
     +         if (o == NULL || checkType(c,o,OBJ_STREAM)) return;
          s = o->ptr;
          grpname = c->argv[3]->ptr;
     ```

     see poc.bpf.c and poc.json.

6. after patch, run the command to install:

     ```console
     $ sudo build/tools/cli/bpftime-cli workloads/ebpf-patch-dev/poc4-redis/poc.json
     Successfully injected. ID: 1
     ```

     results in:

     ```console
     find and load program: xgroupCommand
     load insn cnt: 79
     attach replace 0x562b4fcf7c80
     Successfully attached
     xgroupCommand: 0x562b50b15c98
     c->argc >= 4lookupKeyWriteOrReply: 0x562b50b15c98 0x562b50b23ed8 (nil)
     find func lookupKeyWrite at 545
     ----------------------op code: 0 ret: 0
     xgroupCommand: 0x562b50b15c98
     c->argc >= 4lookupKeyWriteOrReply: 0x562b50b15c98 0x562b50b220d8 (nil)
     find func lookupKeyWrite at 545
     ----------------------op code: 0 ret: 0
     ```
