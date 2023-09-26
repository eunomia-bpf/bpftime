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

#include <stdint.h>
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

#if !defined(_countof)
#define _countof(array) (sizeof(array) / sizeof(array[0]))
#endif

/* Special values for target_pc in struct jump */
#define TARGET_PC_EXIT -1
#define TARGET_PC_DIV_BY_ZERO -2

static void
muldivmod(struct jit_state* state, uint8_t opcode, int src, int dst, int32_t imm);

#define REGISTER_MAP_SIZE 11

/*
 * There are two common x86-64 calling conventions, as discussed at
 * https://en.wikipedia.org/wiki/X86_calling_conventions#x86-64_calling_conventions
 */

#if defined(_WIN32)
static int platform_nonvolatile_registers[] = {RBP, RBX, RDI, RSI, R12, R13, R14, R15};
static int platform_parameter_registers[] = {RCX, RDX, R8, R9};
#define RCX_ALT R10
// Register assignments:
// BPF R0-R4 are "volatile"
// BPF R5-R10 are "non-volatile"
// Map BPF volatile registers to x64 volatile and map BPF non-volatile to
// x64 non-volatile.
// Avoid R12 as we don't support encoding modrm modifier for using R12.
static int register_map[REGISTER_MAP_SIZE] = {
    RAX,
    R10,
    RDX,
    R8,
    R9,
    R14,
    R15,
    RDI,
    RSI,
    RBX,
    RBP,
};
#else
#define RCX_ALT R9
static int platform_nonvolatile_registers[] = {RBP, RBX, R13, R14, R15};
static int platform_parameter_registers[] = {RDI, RSI, RDX, RCX, R8, R9};
static int register_map[REGISTER_MAP_SIZE] = {
    RAX,
    RDI,
    RSI,
    RDX,
    R9,
    R8,
    RBX,
    R13,
    R14,
    R15,
    RBP,
};
#endif

/* Return the x86 register for the given eBPF register */
static int
map_register(int r)
{
    assert(r < REGISTER_MAP_SIZE);
    return register_map[r % REGISTER_MAP_SIZE];
}

/* For testing, this changes the mapping between x86 and eBPF registers */
void
ebpf_set_register_offset(int x)
{
    int i;
    if (x < REGISTER_MAP_SIZE) {
        int tmp[REGISTER_MAP_SIZE];
        memcpy(tmp, register_map, sizeof(register_map));
        for (i = 0; i < REGISTER_MAP_SIZE; i++) {
            register_map[i] = tmp[(i + x) % REGISTER_MAP_SIZE];
        }
    } else {
        /* Shuffle array */
        unsigned int seed = x;
        for (i = 0; i < REGISTER_MAP_SIZE - 1; i++) {
            int j = i + (rand_r(&seed) % (REGISTER_MAP_SIZE - i));
            int tmp = register_map[j];
            register_map[j] = register_map[i];
            register_map[i] = tmp;
        }
    }
}

static int
translate(struct ebpf_vm* vm, struct jit_state* state, char** errmsg)
{
    /* Save platform non-volatile registers */
    for (size_t i = 0; i < _countof(platform_nonvolatile_registers); i++) {
        emit_push(state, platform_nonvolatile_registers[i]);
    }

    /* Move first platform parameter register into register 1 */
    if (map_register(1) != platform_parameter_registers[0]) {
        emit_mov(state, platform_parameter_registers[0], map_register(1));
    }

    /* Copy stack pointer to R10 */
    emit_mov(state, RSP, map_register(10));

    /* Allocate stack space */
    emit_alu64_imm32(state, 0x81, 5, RSP, EBPF_STACK_SIZE);

    for (size_t i = 0; i < vm->num_insts; i++) {
        
        struct ebpf_inst inst = ebpf_fetch_instruction(vm, i);
        state->pc_locs[i] = state->offset;
	    LOG_DEBUG("%08" PRIu64 "x, [%zd] %d %d %d %d\n", *(uint64_t*)(uintptr_t)&inst, i, 
	       inst.dst_reg, inst.src_reg, inst.off, inst.imm);

        int dst = map_register(inst.dst_reg);
        int src = map_register(inst.src_reg);
        uint32_t target_pc = i + inst.off + 1;

        switch (inst.code) {
        case EBPF_OP_ADD_IMM:
            emit_alu32_imm32(state, 0x81, 0, dst, inst.imm);
            break;
        case EBPF_OP_ADD_REG:
            emit_alu32(state, 0x01, src, dst);
            break;
        case EBPF_OP_SUB_IMM:
            emit_alu32_imm32(state, 0x81, 5, dst, inst.imm);
            break;
        case EBPF_OP_SUB_REG:
            emit_alu32(state, 0x29, src, dst);
            break;
        case EBPF_OP_MUL_IMM:
        case EBPF_OP_MUL_REG:
        case EBPF_OP_DIV_IMM:
        case EBPF_OP_DIV_REG:
        case EBPF_OP_MOD_IMM:
        case EBPF_OP_MOD_REG:
            muldivmod(state, inst.code, src, dst, inst.imm);
            break;
        case EBPF_OP_OR_IMM:
            emit_alu32_imm32(state, 0x81, 1, dst, inst.imm);
            break;
        case EBPF_OP_OR_REG:
            emit_alu32(state, 0x09, src, dst);
            break;
        case EBPF_OP_AND_IMM:
            emit_alu32_imm32(state, 0x81, 4, dst, inst.imm);
            break;
        case EBPF_OP_AND_REG:
            emit_alu32(state, 0x21, src, dst);
            break;
        case EBPF_OP_LSH_IMM:
            emit_alu32_imm8(state, 0xc1, 4, dst, inst.imm);
            break;
        case EBPF_OP_LSH_REG:
            emit_mov(state, src, RCX);
            emit_alu32(state, 0xd3, 4, dst);
            break;
        case EBPF_OP_RSH_IMM:
            emit_alu32_imm8(state, 0xc1, 5, dst, inst.imm);
            break;
        case EBPF_OP_RSH_REG:
            emit_mov(state, src, RCX);
            emit_alu32(state, 0xd3, 5, dst);
            break;
        case EBPF_OP_NEG:
            emit_alu32(state, 0xf7, 3, dst);
            break;
        case EBPF_OP_XOR_IMM:
            emit_alu32_imm32(state, 0x81, 6, dst, inst.imm);
            break;
        case EBPF_OP_XOR_REG:
            emit_alu32(state, 0x31, src, dst);
            break;
        case EBPF_OP_MOV_IMM:
            emit_alu32_imm32(state, 0xc7, 0, dst, inst.imm);
            break;
        case EBPF_OP_MOV_REG:
            emit_mov(state, src, dst);
            break;
        case EBPF_OP_ARSH_IMM:
            emit_alu32_imm8(state, 0xc1, 7, dst, inst.imm);
            break;
        case EBPF_OP_ARSH_REG:
            emit_mov(state, src, RCX);
            emit_alu32(state, 0xd3, 7, dst);
            break;

        case EBPF_OP_LE:
            /* No-op */
            break;
        case EBPF_OP_BE:
            if (inst.imm == 16) {
                /* rol */
                emit1(state, 0x66); /* 16-bit override */
                emit_alu32_imm8(state, 0xc1, 0, dst, 8);
                /* and */
                emit_alu32_imm32(state, 0x81, 4, dst, 0xffff);
            } else if (inst.imm == 32 || inst.imm == 64) {
                /* bswap */
                emit_basic_rex(state, inst.imm == 64, 0, dst);
                emit1(state, 0x0f);
                emit1(state, 0xc8 | (dst & 7));
            }
            break;

        case EBPF_OP_ADD64_IMM:
            emit_alu64_imm32(state, 0x81, 0, dst, inst.imm);
            break;
        case EBPF_OP_ADD64_REG:
            emit_alu64(state, 0x01, src, dst);
            break;
        case EBPF_OP_SUB64_IMM:
            emit_alu64_imm32(state, 0x81, 5, dst, inst.imm);
            break;
        case EBPF_OP_SUB64_REG:
            emit_alu64(state, 0x29, src, dst);
            break;
        case EBPF_OP_MUL64_IMM:
        case EBPF_OP_MUL64_REG:
        case EBPF_OP_DIV64_IMM:
        case EBPF_OP_DIV64_REG:
        case EBPF_OP_MOD64_IMM:
        case EBPF_OP_MOD64_REG:
            muldivmod(state, inst.code, src, dst, inst.imm);
            break;
        case EBPF_OP_OR64_IMM:
            emit_alu64_imm32(state, 0x81, 1, dst, inst.imm);
            break;
        case EBPF_OP_OR64_REG:
            emit_alu64(state, 0x09, src, dst);
            break;
        case EBPF_OP_AND64_IMM:
            emit_alu64_imm32(state, 0x81, 4, dst, inst.imm);
            break;
        case EBPF_OP_AND64_REG:
            emit_alu64(state, 0x21, src, dst);
            break;
        case EBPF_OP_LSH64_IMM:
            emit_alu64_imm8(state, 0xc1, 4, dst, inst.imm);
            break;
        case EBPF_OP_LSH64_REG:
            emit_mov(state, src, RCX);
            emit_alu64(state, 0xd3, 4, dst);
            break;
        case EBPF_OP_RSH64_IMM:
            emit_alu64_imm8(state, 0xc1, 5, dst, inst.imm);
            break;
        case EBPF_OP_RSH64_REG:
            emit_mov(state, src, RCX);
            emit_alu64(state, 0xd3, 5, dst);
            break;
        case EBPF_OP_NEG64:
            emit_alu64(state, 0xf7, 3, dst);
            break;
        case EBPF_OP_XOR64_IMM:
            emit_alu64_imm32(state, 0x81, 6, dst, inst.imm);
            break;
        case EBPF_OP_XOR64_REG:
            emit_alu64(state, 0x31, src, dst);
            break;
        case EBPF_OP_MOV64_IMM:
            emit_load_imm(state, dst, inst.imm);
            break;
        case EBPF_OP_MOV64_REG:
            emit_mov(state, src, dst);
            break;
        case EBPF_OP_ARSH64_IMM:
            emit_alu64_imm8(state, 0xc1, 7, dst, inst.imm);
            break;
        case EBPF_OP_ARSH64_REG:
            emit_mov(state, src, RCX);
            emit_alu64(state, 0xd3, 7, dst);
            break;

        /* TODO use 8 bit immediate when possible */
        case EBPF_OP_JA:
            emit_jmp(state, target_pc);
            break;
        case EBPF_OP_JEQ_IMM:
            emit_cmp_imm32(state, dst, inst.imm);
            emit_jcc(state, 0x84, target_pc);
            break;
        case EBPF_OP_JEQ_REG:
            emit_cmp(state, src, dst);
            emit_jcc(state, 0x84, target_pc);
            break;
        case EBPF_OP_JGT_IMM:
            emit_cmp_imm32(state, dst, inst.imm);
            emit_jcc(state, 0x87, target_pc);
            break;
        case EBPF_OP_JGT_REG:
            emit_cmp(state, src, dst);
            emit_jcc(state, 0x87, target_pc);
            break;
        case EBPF_OP_JGE_IMM:
            emit_cmp_imm32(state, dst, inst.imm);
            emit_jcc(state, 0x83, target_pc);
            break;
        case EBPF_OP_JGE_REG:
            emit_cmp(state, src, dst);
            emit_jcc(state, 0x83, target_pc);
            break;
        case EBPF_OP_JLT_IMM:
            emit_cmp_imm32(state, dst, inst.imm);
            emit_jcc(state, 0x82, target_pc);
            break;
        case EBPF_OP_JLT_REG:
            emit_cmp(state, src, dst);
            emit_jcc(state, 0x82, target_pc);
            break;
        case EBPF_OP_JLE_IMM:
            emit_cmp_imm32(state, dst, inst.imm);
            emit_jcc(state, 0x86, target_pc);
            break;
        case EBPF_OP_JLE_REG:
            emit_cmp(state, src, dst);
            emit_jcc(state, 0x86, target_pc);
            break;
        case EBPF_OP_JSET_IMM:
            emit_alu64_imm32(state, 0xf7, 0, dst, inst.imm);
            emit_jcc(state, 0x85, target_pc);
            break;
        case EBPF_OP_JSET_REG:
            emit_alu64(state, 0x85, src, dst);
            emit_jcc(state, 0x85, target_pc);
            break;
        case EBPF_OP_JNE_IMM:
            emit_cmp_imm32(state, dst, inst.imm);
            emit_jcc(state, 0x85, target_pc);
            break;
        case EBPF_OP_JNE_REG:
            emit_cmp(state, src, dst);
            emit_jcc(state, 0x85, target_pc);
            break;
        case EBPF_OP_JSGT_IMM:
            emit_cmp_imm32(state, dst, inst.imm);
            emit_jcc(state, 0x8f, target_pc);
            break;
        case EBPF_OP_JSGT_REG:
            emit_cmp(state, src, dst);
            emit_jcc(state, 0x8f, target_pc);
            break;
        case EBPF_OP_JSGE_IMM:
            emit_cmp_imm32(state, dst, inst.imm);
            emit_jcc(state, 0x8d, target_pc);
            break;
        case EBPF_OP_JSGE_REG:
            emit_cmp(state, src, dst);
            emit_jcc(state, 0x8d, target_pc);
            break;
        case EBPF_OP_JSLT_IMM:
            emit_cmp_imm32(state, dst, inst.imm);
            emit_jcc(state, 0x8c, target_pc);
            break;
        case EBPF_OP_JSLT_REG:
            emit_cmp(state, src, dst);
            emit_jcc(state, 0x8c, target_pc);
            break;
        case EBPF_OP_JSLE_IMM:
            emit_cmp_imm32(state, dst, inst.imm);
            emit_jcc(state, 0x8e, target_pc);
            break;
        case EBPF_OP_JSLE_REG:
            emit_cmp(state, src, dst);
            emit_jcc(state, 0x8e, target_pc);
            break;
        case EBPF_OP_JEQ32_IMM:
            emit_cmp32_imm32(state, dst, inst.imm);
            emit_jcc(state, 0x84, target_pc);
            break;
        case EBPF_OP_JEQ32_REG:
            emit_cmp32(state, src, dst);
            emit_jcc(state, 0x84, target_pc);
            break;
        case EBPF_OP_JGT32_IMM:
            emit_cmp32_imm32(state, dst, inst.imm);
            emit_jcc(state, 0x87, target_pc);
            break;
        case EBPF_OP_JGT32_REG:
            emit_cmp32(state, src, dst);
            emit_jcc(state, 0x87, target_pc);
            break;
        case EBPF_OP_JGE32_IMM:
            emit_cmp32_imm32(state, dst, inst.imm);
            emit_jcc(state, 0x83, target_pc);
            break;
        case EBPF_OP_JGE32_REG:
            emit_cmp32(state, src, dst);
            emit_jcc(state, 0x83, target_pc);
            break;
        case EBPF_OP_JLT32_IMM:
            emit_cmp32_imm32(state, dst, inst.imm);
            emit_jcc(state, 0x82, target_pc);
            break;
        case EBPF_OP_JLT32_REG:
            emit_cmp32(state, src, dst);
            emit_jcc(state, 0x82, target_pc);
            break;
        case EBPF_OP_JLE32_IMM:
            emit_cmp32_imm32(state, dst, inst.imm);
            emit_jcc(state, 0x86, target_pc);
            break;
        case EBPF_OP_JLE32_REG:
            emit_cmp32(state, src, dst);
            emit_jcc(state, 0x86, target_pc);
            break;
        case EBPF_OP_JSET32_IMM:
            emit_alu32_imm32(state, 0xf7, 0, dst, inst.imm);
            emit_jcc(state, 0x85, target_pc);
            break;
        case EBPF_OP_JSET32_REG:
            emit_alu32(state, 0x85, src, dst);
            emit_jcc(state, 0x85, target_pc);
            break;
        case EBPF_OP_JNE32_IMM:
            emit_cmp32_imm32(state, dst, inst.imm);
            emit_jcc(state, 0x85, target_pc);
            break;
        case EBPF_OP_JNE32_REG:
            emit_cmp32(state, src, dst);
            emit_jcc(state, 0x85, target_pc);
            break;
        case EBPF_OP_JSGT32_IMM:
            emit_cmp32_imm32(state, dst, inst.imm);
            emit_jcc(state, 0x8f, target_pc);
            break;
        case EBPF_OP_JSGT32_REG:
            emit_cmp32(state, src, dst);
            emit_jcc(state, 0x8f, target_pc);
            break;
        case EBPF_OP_JSGE32_IMM:
            emit_cmp32_imm32(state, dst, inst.imm);
            emit_jcc(state, 0x8d, target_pc);
            break;
        case EBPF_OP_JSGE32_REG:
            emit_cmp32(state, src, dst);
            emit_jcc(state, 0x8d, target_pc);
            break;
        case EBPF_OP_JSLT32_IMM:
            emit_cmp32_imm32(state, dst, inst.imm);
            emit_jcc(state, 0x8c, target_pc);
            break;
        case EBPF_OP_JSLT32_REG:
            emit_cmp32(state, src, dst);
            emit_jcc(state, 0x8c, target_pc);
            break;
        case EBPF_OP_JSLE32_IMM:
            emit_cmp32_imm32(state, dst, inst.imm);
            emit_jcc(state, 0x8e, target_pc);
            break;
        case EBPF_OP_JSLE32_REG:
            emit_cmp32(state, src, dst);
            emit_jcc(state, 0x8e, target_pc);
            break;
        case EBPF_OP_CALL:
            /* We reserve RCX for shifts */
            emit_mov(state, RCX_ALT, RCX);
            emit_call(state, vm->ext_funcs[inst.imm]);
            if (inst.imm == vm->unwind_stack_extension_index) {
                emit_cmp_imm32(state, map_register(0), 0);
                emit_jcc(state, 0x84, TARGET_PC_EXIT);
            }
            break;
        case EBPF_OP_EXIT:
            if ((int)i != vm->num_insts - 1) {
                emit_jmp(state, TARGET_PC_EXIT);
            }
            break;

        case EBPF_OP_LDXW:
            emit_load(state, S32, src, dst, inst.off);
            break;
        case EBPF_OP_LDXH:
            emit_load(state, S16, src, dst, inst.off);
            break;
        case EBPF_OP_LDXB:
            emit_load(state, S8, src, dst, inst.off);
            break;
        case EBPF_OP_LDXDW:
            emit_load(state, S64, src, dst, inst.off);
            break;

        case EBPF_OP_STW:
            emit_store_imm32(state, S32, dst, inst.off, inst.imm);
            break;
        case EBPF_OP_STH:
            emit_store_imm32(state, S16, dst, inst.off, inst.imm);
            break;
        case EBPF_OP_STB:
            emit_store_imm32(state, S8, dst, inst.off, inst.imm);
            break;
        case EBPF_OP_STDW:
            emit_store_imm32(state, S64, dst, inst.off, inst.imm);
            break;

        case EBPF_OP_STXW:
            emit_store(state, S32, src, dst, inst.off);
            break;
        case EBPF_OP_STXH:
            emit_store(state, S16, src, dst, inst.off);
            break;
        case EBPF_OP_STXB:
            emit_store(state, S8, src, dst, inst.off);
            break;
        case EBPF_OP_STXDW:
            emit_store(state, S64, src, dst, inst.off);
            break;

        case EBPF_OP_LDDW: {
            struct ebpf_inst inst2 = ebpf_fetch_instruction(vm, ++i);
            uint64_t imm = (uint32_t)inst.imm | ((uint64_t)inst2.imm << 32);
            emit_load_imm(state, dst, imm);
            break;
        }

        default:
            *errmsg = ebpf_error("Unknown instruction at PC %d: opcode %02x", i, inst.code);
            return -1;
        }
    }

    /* Epilogue */
    state->exit_loc = state->offset;

    /* Move register 0 into rax */
    if (map_register(0) != RAX) {
        emit_mov(state, map_register(0), RAX);
    }

    /* Deallocate stack space */
    emit_alu64_imm32(state, 0x81, 0, RSP, EBPF_STACK_SIZE);

    /* Restore platform non-volatile registers */
    for (size_t i = 0; i < _countof(platform_nonvolatile_registers); i++) {
        emit_pop(state, platform_nonvolatile_registers[_countof(platform_nonvolatile_registers) - i - 1]);
    }

    emit1(state, 0xc3); /* ret */

    return 0;
}

static void
muldivmod(struct jit_state* state, uint8_t opcode, int src, int dst, int32_t imm)
{
    bool mul = (opcode & EBPF_ALU_OP_MASK) == (EBPF_OP_MUL_IMM & EBPF_ALU_OP_MASK);
    bool div = (opcode & EBPF_ALU_OP_MASK) == (EBPF_OP_DIV_IMM & EBPF_ALU_OP_MASK);
    bool mod = (opcode & EBPF_ALU_OP_MASK) == (EBPF_OP_MOD_IMM & EBPF_ALU_OP_MASK);
    bool is64 = (opcode & EBPF_CLS_MASK) == EBPF_CLS_ALU64;
    bool reg = (opcode & EBPF_SRC_REG) == EBPF_SRC_REG;

    // Short circuit for imm == 0.
    if (!reg && imm == 0) {
        if (div || mul) {
            // For division and multiplication, set result to zero.
            emit_alu32(state, 0x31, dst, dst);
        } else {
            // For modulo, set result to dividend.
            emit_mov(state, dst, dst);
        }
        return;
    }

    if (dst != RAX) {
        emit_push(state, RAX);
    }

    if (dst != RDX) {
        emit_push(state, RDX);
    }

    // Load the divisor into RCX.
    if (imm) {
        emit_load_imm(state, RCX, imm);
    } else {
        emit_mov(state, src, RCX);
    }

    // Load the dividend into RAX.
    emit_mov(state, dst, RAX);

    // BPF has two different semantics for division and modulus. For division
    // if the divisor is zero, the result is zero.  For modulus, if the divisor
    // is zero, the result is the dividend. To handle this we set the divisor
    // to 1 if it is zero and then set the result to zero if the divisor was
    // zero (for division) or set the result to the dividend if the divisor was
    // zero (for modulo).

    if (div || mod) {
        // Check if divisor is zero.
        if (is64) {
            emit_alu64(state, 0x85, RCX, RCX);
        } else {
            emit_alu32(state, 0x85, RCX, RCX);
        }

        // Save the dividend for the modulo case.
        if (mod) {
            emit_push(state, RAX); // Save dividend.
        }

        // Save the result of the test.
        emit1(state, 0x9c); /* pushfq */

        // Set the divisor to 1 if it is zero.
        emit_load_imm(state, RDX, 1);
        emit1(state, 0x48);
        emit1(state, 0x0f);
        emit1(state, 0x44);
        emit1(state, 0xca); /* cmove rcx,rdx */

        /* xor %edx,%edx */
        emit_alu32(state, 0x31, RDX, RDX);
    }

    if (is64) {
        emit_rex(state, 1, 0, 0, 0);
    }

    // Multiply or divide.
    emit_alu32(state, 0xf7, mul ? 4 : 6, RCX);

    // Division operation stores the remainder in RDX and the quotient in RAX.
    if (div || mod) {
        // Restore the result of the test.
        emit1(state, 0x9d); /* popfq */

        // If zero flag is set, then the divisor was zero.

        if (div) {
            // Set the dividend to zero if the divisor was zero.
            emit_load_imm(state, RCX, 0);

            // Store 0 in RAX if the divisor was zero.
            // Use conditional move to avoid a branch.
            emit1(state, 0x48);
            emit1(state, 0x0f);
            emit1(state, 0x44);
            emit1(state, 0xc1); /* cmove rax,rcx */
        } else {
            // Restore dividend to RCX.
            emit_pop(state, RCX);

            // Store the dividend in RAX if the divisor was zero.
            // Use conditional move to avoid a branch.
            emit1(state, 0x48);
            emit1(state, 0x0f);
            emit1(state, 0x44);
            emit1(state, 0xd1); /* cmove rdx,rcx */
        }
    }

    if (dst != RDX) {
        if (mod) {
            emit_mov(state, RDX, dst);
        }
        emit_pop(state, RDX);
    }
    if (dst != RAX) {
        if (div || mul) {
            emit_mov(state, RAX, dst);
        }
        emit_pop(state, RAX);
    }
}

static void
resolve_jumps(struct jit_state* state)
{
    int i;
    for (i = 0; i < state->num_jumps; i++) {
        struct jump jump = state->jumps[i];

        int target_loc;
        if ((int)jump.target_pc == TARGET_PC_EXIT) {
            target_loc = state->exit_loc;
        } else if ((int)jump.target_pc == TARGET_PC_DIV_BY_ZERO) {
            target_loc = state->div_by_zero_loc;
        } else {
            target_loc = state->pc_locs[jump.target_pc];
        }

        /* Assumes jump offset is at end of instruction */
        uint32_t rel = target_loc - (jump.offset_loc + sizeof(uint32_t));

        uint8_t* offset_ptr = &state->buf[jump.offset_loc];
        memcpy(offset_ptr, &rel, sizeof(uint32_t));
    }
}

int
ebpf_translate_x86_64(struct ebpf_vm* vm, uint8_t* buffer, size_t* size, char** errmsg)
{
    struct jit_state state;
    int result = -1;

    state.offset = 0;
    state.size = *size;
    state.buf = buffer;
    state.pc_locs = calloc(EBPF_MAX_INSTS + 1, sizeof(state.pc_locs[0]));
    state.jumps = calloc(EBPF_MAX_INSTS, sizeof(state.jumps[0]));
    state.num_jumps = 0;

    if (translate(vm, &state, errmsg) < 0) {
        goto out;
    }

    if (state.num_jumps == EBPF_MAX_INSTS) {
        *errmsg = ebpf_error("Excessive number of jump targets");
        goto out;
    }

    if (state.offset == state.size) {
        *errmsg = ebpf_error("Target buffer too small");
        goto out;
    }

    resolve_jumps(&state);
    result = 0;

    *size = state.offset;

out:
    free(state.pc_locs);
    free(state.jumps);
    return result;
}
