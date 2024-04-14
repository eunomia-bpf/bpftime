/*
 * Copyright 2015 Big Switch Networks, Inc
 * Copyright 2017 Google Inc.
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
#include <inttypes.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <math.h>
#include "ebpf-vm.h"

// void
// ebpf_set_register_offset(int x);
static void*
readfile(const char* path, size_t maxlen, size_t* len);
static void
register_functions(struct ebpf_vm* vm);

static void
usage(const char* name)
{
    fprintf(stderr, "usage: %s [-h] [-j|--jit] [-m|--mem PATH] BINARY\n", name);
    fprintf(stderr, "\nExecutes the eBPF code in BINARY and prints the result to stdout.\n");
    fprintf(
        stderr, "If --mem is given then the specified file will be read and a pointer\nto its data passed in r1.\n");
    fprintf(stderr, "If --jit is given then the JIT compiler will be used.\n");
    fprintf(stderr, "\nOther options:\n");
    fprintf(stderr, "  -r, --register-offset NUM: Change the mapping from eBPF to x86 registers\n");
    fprintf(stderr, "  -U, --unload: unload the code and reload it (for testing only)\n");
    fprintf(
        stderr, "  -R, --reload: reload the code, without unloading it first (for testing only, this should fail)\n");
}

int
main(int argc, char** argv)
{
    struct option longopts[] = {
        {
            .name = "help",
            .val = 'h',
        },
        {.name = "mem", .val = 'm', .has_arg = 1},
        {.name = "jit", .val = 'j'},
        {.name = "register-offset", .val = 'r', .has_arg = 1},
        {.name = "unload", .val = 'U'}, /* for unit test only */
        {.name = "reload", .val = 'R'}, /* for unit test only */
        {0}};

    const char* mem_filename = NULL;
    bool jit = false;
    bool unload = false;
    bool reload = false;

    uint64_t secret = (uint64_t)rand() << 32 | (uint64_t)rand();

    int opt;
    while ((opt = getopt_long(argc, argv, "hm:jr:UR", longopts, NULL)) != -1) {
        switch (opt) {
        case 'm':
            mem_filename = optarg;
            break;
        case 'j':
            jit = true;
            break;
        case 'r':
#if defined(__x86_64__) || defined(_M_X64)
            // ebpf_set_register_offset(atoi(optarg));
#endif
            break;
        case 'h':
            usage(argv[0]);
            return 0;
        case 'U':
            unload = true;
            break;
        case 'R':
            reload = true;
            break;
        default:
            usage(argv[0]);
            return 0;
        }
    }

    if (unload && reload) {
        fprintf(stderr, "-U and -R can not be used together\n");
        return 1;
    }

    if (argc != optind + 1) {
        usage(argv[0]);
        return 0;
    }

    const char* code_filename = argv[optind];
    size_t code_len;
    void* code = readfile(code_filename, 1024 * 1024, &code_len);
    if (code == NULL) {
        return 1;
    }

    size_t mem_len = 0;
    void* mem = NULL;
    if (mem_filename != NULL) {
        mem = readfile(mem_filename, 1024 * 1024, &mem_len);
        if (mem == NULL) {
            return 1;
        }
    }

    struct ebpf_vm* vm = ebpf_create();
    if (!vm) {
        fprintf(stderr, "Failed to create VM\n");
        return 1;
    }

    if (ebpf_set_pointer_secret(vm, secret) != 0) {
        fprintf(stderr, "Failed to set pointer secret\n");
        return 1;
    }

    register_functions(vm);

    /*
     * The ELF magic corresponds to an RSH instruction with an offset,
     * which is invalid.
     */
#if defined(UBPF_HAS_ELF_H)
    bool elf = code_len >= SELFMAG && !memcmp(code, ELFMAG, SELFMAG);
#endif

    char* errmsg;
    int rv;
load:
#if defined(UBPF_HAS_ELF_H)
    if (elf) {
        rv = ebpf_load_elf(vm, code, code_len, &errmsg);
    } else {
#endif
        rv = ebpf_load(vm, code, code_len, &errmsg);
#if defined(UBPF_HAS_ELF_H)
    }
#endif
    if (unload) {
        ebpf_unload_code(vm);
        unload = false;
        goto load;
    }
    if (reload) {
        reload = false;
        goto load;
    }

    free(code);

    if (rv < 0) {
        fprintf(stderr, "Failed to load code: %s\n", errmsg);
        free(errmsg);
        ebpf_destroy(vm);
        return 1;
    }

    uint64_t ret;

    if (jit) {
        ebpf_jit_fn fn = ebpf_compile(vm, &errmsg);
        if (fn == NULL) {
            fprintf(stderr, "Failed to compile: %s\n", errmsg);
            free(errmsg);
            free(mem);
            return 1;
        }
        ret = fn(mem, mem_len);
    } else {
        if (ebpf_exec(vm, mem, mem_len, &ret) < 0)
            ret = UINT64_MAX;
    }

    printf("0x%" PRIx64 "\n", ret);

    ebpf_destroy(vm);
    free(mem);

    return 0;
}

static void*
readfile(const char* path, size_t maxlen, size_t* len)
{
    FILE* file;
    if (!strcmp(path, "-")) {
        file = fdopen(STDIN_FILENO, "r");
    } else {
        file = fopen(path, "r");
    }

    if (file == NULL) {
        fprintf(stderr, "Failed to open %s: %s\n", path, strerror(errno));
        return NULL;
    }

    char* data = calloc(maxlen, 1);
    size_t offset = 0;
    size_t rv;
    while ((rv = fread(data + offset, 1, maxlen - offset, file)) > 0) {
        offset += rv;
    }

    if (ferror(file)) {
        fprintf(stderr, "Failed to read %s: %s\n", path, strerror(errno));
        fclose(file);
        free(data);
        return NULL;
    }

    if (!feof(file)) {
        fprintf(stderr, "Failed to read %s because it is too large (max %u bytes)\n", path, (unsigned)maxlen);
        fclose(file);
        free(data);
        return NULL;
    }

    fclose(file);
    if (len) {
        *len = offset;
    }
    return (void*)data;
}

uint64_t
memfrob_ext(uint64_t s, uint64_t n)
{
    size_t p1 = s;
    for (uint64_t i = 0; i < n; i++) {
        ((char*)p1)[i] ^= 42;
    }
    return s;
}

static uint64_t
gather_bytes(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e)
{
    return (((uint64_t)a) << (uint64_t)32) | (((uint32_t)b) << (uint64_t)24) | (((uint32_t)c) << (uint64_t)16) | (((uint16_t)d) << (uint64_t)8) | (uint64_t)e;
}

static void
trash_registers(void)
{
    /* Overwrite all caller-save registers */
#if __x86_64__
    asm("mov $0xf0, %rax;"
        "mov $0xf1, %rcx;"
        "mov $0xf2, %rdx;"
        "mov $0xf3, %rsi;"
        "mov $0xf4, %rdi;"
        "mov $0xf5, %r8;"
        "mov $0xf6, %r9;"
        "mov $0xf7, %r10;"
        "mov $0xf8, %r11;");
#elif __aarch64__
    asm("mov w0, #0xf0;"
        "mov w1, #0xf1;"
        "mov w2, #0xf2;"
        "mov w3, #0xf3;"
        "mov w4, #0xf4;"
        "mov w5, #0xf5;"
        "mov w6, #0xf6;"
        "mov w7, #0xf7;"
        "mov w8, #0xf8;"
        "mov w9, #0xf9;"
        "mov w10, #0xfa;"
        "mov w11, #0xfb;"
        "mov w12, #0xfc;"
        "mov w13, #0xfd;"
        "mov w14, #0xfe;"
        "mov w15, #0xff;" ::
            : "w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9", "w10", "w11", "w12", "w13", "w14", "w15");
#elif __arm__
    // implementation for arm32 architecture
    asm("mov r0, #0xf0;"
        "mov r1, #0xf1;"
        "mov r2, #0xf2;"
        "mov r3, #0xf3;"
        "mov r4, #0xf4;"
        "mov r5, #0xf5;"
        "mov r6, #0xf6;"
        "mov r7, #0xf7;"
        "mov r8, #0xf8;"
        "mov r9, #0xf9;"
    );
#else
    fprintf(stderr, "trash_registers not implemented for this architecture.\n");
    exit(1);
#endif
}

static uint32_t
sqrti(uint32_t x)
{
    return sqrt(x);
}

static uint64_t
unwind(uint64_t i)
{
    return i;
}

static uint64_t
strcmp_ext(uint64_t a, uint64_t b) {
    size_t p1 = a;
    size_t p2 = b;
    return strcmp((const char *)p1, (const char *)p2);
}

static void
register_functions(struct ebpf_vm* vm)
{
    ebpf_register(vm, 0, "gather_bytes", gather_bytes);
    ebpf_register(vm, 1, "memfrob", memfrob_ext);
    ebpf_register(vm, 2, "trash_registers", trash_registers);
    ebpf_register(vm, 3, "sqrti", sqrti);
    ebpf_register(vm, 4, "strcmp_ext", strcmp_ext);
    ebpf_register(vm, 5, "unwind", unwind);
    ebpf_set_unwind_function_index(vm, 5);
}
