from .asm_parser import parse, Reg, Imm, MemRef
import struct
try:
    from StringIO import StringIO as io
except ImportError:
    from io import BytesIO as io

Inst = struct.Struct("BBHI")

MEM_SIZES = {
    'w': 0,
    'h': 1,
    'b': 2,
    'dw': 3,
}

MEM_LOAD_OPS = { 'ldx' + k: (0x61 | (v << 3)) for k, v in list(MEM_SIZES.items()) }
MEM_STORE_IMM_OPS = { 'st' + k: (0x62 | (v << 3))  for k, v in list(MEM_SIZES.items()) }
MEM_STORE_REG_OPS = { 'stx' + k: (0x63 | (v << 3)) for k, v in list(MEM_SIZES.items()) }

UNARY_ALU_OPS = {
    'neg': 8,
}

BINARY_ALU_OPS = {
    'add': 0,
    'sub': 1,
    'mul': 2,
    'div': 3,
    'or': 4,
    'and': 5,
    'lsh': 6,
    'rsh': 7,
    'mod': 9,
    'xor': 10,
    'mov': 11,
    'arsh': 12,
}

UNARY_ALU32_OPS = { k + '32': v for k, v in list(UNARY_ALU_OPS.items()) }
BINARY_ALU32_OPS = { k + '32': v for k, v in list(BINARY_ALU_OPS.items()) }

END_OPS = {
    'le16': (0xd4, 16),
    'le32': (0xd4, 32),
    'le64': (0xd4, 64),
    'be16': (0xdc, 16),
    'be32': (0xdc, 32),
    'be64': (0xdc, 64),
}

JMP_CMP_OPS = {
    'jeq': 1,
    'jgt': 2,
    'jge': 3,
    'jset': 4,
    'jne': 5,
    'jsgt': 6,
    'jsge': 7,
    'jlt': 10,
    'jle': 11,
    'jslt': 12,
    'jsle': 13,
}

JMP_MISC_OPS = {
    'ja': 0,
    'call': 8,
    'exit': 9,
}

def pack(opcode, dst, src, offset, imm):
    return Inst.pack(opcode & 0xff, (dst | (src << 4)) & 0xff, offset & 0xffff, imm & 0xffffffff)

def assemble_binop(op, cls, ops, dst, src, offset):
    opcode = cls | (ops[op] << 4)
    if isinstance(src, Imm):
        return pack(opcode, dst.num, 0, offset, src.value)
    else:
        return pack(opcode | 0x08, dst.num, src.num, offset, 0)

def assemble_one(inst):
    op = inst[0]
    if op in MEM_LOAD_OPS:
        opcode = MEM_LOAD_OPS[op]
        return pack(opcode, inst[1].num, inst[2].reg.num, inst[2].offset, 0)
    elif op == "lddw":
        a = pack(0x18, inst[1].num, 0, 0, inst[2].value)
        b = pack(0, 0, 0, 0, inst[2].value >> 32)
        return a + b
    elif op in MEM_STORE_IMM_OPS:
        opcode = MEM_STORE_IMM_OPS[op]
        return pack(opcode, inst[1].reg.num, 0, inst[1].offset, inst[2].value)
    elif op in MEM_STORE_REG_OPS:
        opcode = MEM_STORE_REG_OPS[op]
        return pack(opcode, inst[1].reg.num, inst[2].num, inst[1].offset, 0)
    elif op in UNARY_ALU_OPS:
        opcode = 0x07 | (UNARY_ALU_OPS[op] << 4)
        return pack(opcode, inst[1].num, 0, 0, 0)
    elif op in UNARY_ALU32_OPS:
        opcode = 0x04 | (UNARY_ALU32_OPS[op] << 4)
        return pack(opcode, inst[1].num, 0, 0, 0)
    elif op in BINARY_ALU_OPS:
        return assemble_binop(op, 0x07, BINARY_ALU_OPS, inst[1], inst[2], 0)
    elif op in BINARY_ALU32_OPS:
        return assemble_binop(op, 0x04, BINARY_ALU32_OPS, inst[1], inst[2], 0)
    elif op in END_OPS:
        opcode, imm = END_OPS[op]
        return pack(opcode, inst[1].num, 0, 0, imm)
    elif op in JMP_CMP_OPS:
        return assemble_binop(op, 0x05, JMP_CMP_OPS, inst[1], inst[2], inst[3])
    elif op in JMP_MISC_OPS:
        opcode = 0x05 | (JMP_MISC_OPS[op] << 4)
        if op == 'ja':
            return pack(opcode, 0, 0, inst[1], 0)
        elif op == 'call':
            return pack(opcode, 0, 0, 0, inst[1].value)
        elif op == 'exit':
            return pack(opcode, 0, 0, 0, 0)
    else:
        raise ValueError("unexpected instruction %r" % op)

def assemble(source):
    insts = parse(source)
    output = io()
    for inst in insts:
        output.write(assemble_one(inst))
    return output.getvalue()
