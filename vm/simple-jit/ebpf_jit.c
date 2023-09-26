/*
 * Copyright 2015 Big Switch Networks, Inc
 * Copyright 2017 Google Inc.
 * Copyright 2022 Linaro Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <errno.h>
#include <assert.h>
#include "ebpf_inst.h"
#include "ebpf_jit_x86_64.h"

#define UNUSED(x) ((void)x)

int
ebpf_translate(struct ebpf_vm* vm, uint8_t* buffer, size_t* size, char** errmsg)
{
    return vm->translate(vm, buffer, size, errmsg);
}

int
ebpf_translate_null(struct ebpf_vm* vm, uint8_t* buffer, size_t* size, char** errmsg)
{
    /* NULL JIT target - just returns an error. */
    UNUSED(vm);
    UNUSED(buffer);
    UNUSED(size);
    *errmsg = ebpf_error("Code can not be JITed on this target.");
    return -1;
}

ebpf_jit_fn
ebpf_compile(struct ebpf_vm* vm, char** errmsg)
{
    void* jitted = NULL;
    uint8_t* buffer = NULL;
    size_t jitted_size;

    if (vm->jitted_function) {
        return vm->jitted_function;
    }

    vm->jitted_function = NULL;
    *errmsg = NULL;

    if (!vm->insnsi) {
        *errmsg = ebpf_error("code has not been loaded into this VM");
        return NULL;
    }

    jitted_size = 65536;
    buffer = calloc(jitted_size, 1);

    if (ebpf_translate(vm, buffer, &jitted_size, errmsg) < 0) {
        goto out;
    }

    jitted = mmap(0, jitted_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (jitted == MAP_FAILED) {
        *errmsg = ebpf_error("internal ebpf error: mmap failed: %s\n", strerror(errno));
        goto out;
    }

    memcpy(jitted, buffer, jitted_size);

    if (mprotect(jitted, jitted_size, PROT_READ | PROT_EXEC) < 0) {
        *errmsg = ebpf_error("internal ebpf error: mprotect failed: %s\n", strerror(errno));
        goto out;
    }

    vm->jitted_function = jitted;
    vm->jitted_size = jitted_size;

out:
    free(buffer);
    if (jitted && vm->jitted_function == NULL) {
        munmap(jitted, jitted_size);
    }
    return vm->jitted_function;
}
