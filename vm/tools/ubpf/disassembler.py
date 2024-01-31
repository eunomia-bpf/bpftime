import struct
try:
    from StringIO import StringIO as io
except ImportError:
    from io import StringIO as io

Inst = struct.Struct("BBHI")

CLASSES = {
    0: "ld",
    1: "ldx",
    2: "st",
    3: "stx",
    4: "alu",
    5: "jmp",
    7: "alu64",
}

ALU_OPCODES = {
    0: 'add',
    1: 'sub',
    2: 'mul',
    3: 'div',
    4: 'or',
    5: 'and',
    6: 'lsh',
    7: 'rsh',
    8: 'neg',
    9: 'mod',
    10: 'xor',
    11: 'mov',
    12: 'arsh',
    13: '(endian)',
}

JMP_OPCODES = {
    0: 'ja',
    1: 'jeq',
    2: 'jgt',
    3: 'jge',
    4: 'jset',
    5: 'jne',
    6: 'jsgt',
    7: 'jsge',
    8: 'call',
    9: 'exit',
    10: 'jlt',
    11: 'jle',
    12: 'jslt',
    13: 'jsle',
}

MODES = {
    0: 'imm',
    1: 'abs',
    2: 'ind',
    3: 'mem',
    6: 'xadd',
}

SIZES = {
    0: 'w',
    1: 'h',
    2: 'b',
    3: 'dw',
}

BPF_CLASS_LD = 0
BPF_CLASS_LDX = 1
BPF_CLASS_ST = 2
BPF_CLASS_STX = 3
BPF_CLASS_ALU = 4
BPF_CLASS_JMP = 5
BPF_CLASS_ALU64 = 7

BPF_ALU_NEG = 8
BPF_ALU_END = 13

def R(reg):
    return "r" + str(reg)

def I(imm):
    return "%#x" % imm

def M(base, off):
    if off != 0:
        return "[%s%s]" % (base, O(off))
    else:
        return "[%s]" % base

def O(off):
    if off <= 32767:
        return "+" + str(off)
    else:
        return "-" + str(65536-off)

def disassemble_one(data, offset):
    code, regs, off, imm = Inst.unpack_from(data, offset)
    dst_reg = regs & 0xf
    src_reg = (regs >> 4) & 0xf
    cls = code & 7

    class_name = CLASSES.get(cls)

    if cls == BPF_CLASS_ALU or cls == BPF_CLASS_ALU64:
        source = (code >> 3) & 1
        opcode = (code >> 4) & 0xf
        opcode_name = ALU_OPCODES.get(opcode)
        if cls == BPF_CLASS_ALU:
            opcode_name += "32"

        if opcode == BPF_ALU_END:
            opcode_name = source == 1 and "be" or "le"
            return "%s%d %s" % (opcode_name, imm, R(dst_reg))
        elif opcode == BPF_ALU_NEG:
            return "%s %s" % (opcode_name, R(dst_reg))
        elif source == 0:
            return "%s %s, %s" % (opcode_name, R(dst_reg), I(imm))
        else:
            return "%s %s, %s" % (opcode_name, R(dst_reg), R(src_reg))
    elif cls == BPF_CLASS_JMP:
        source = (code >> 3) & 1
        opcode = (code >> 4) & 0xf
        opcode_name = JMP_OPCODES.get(opcode)

        if opcode_name == "exit":
            return opcode_name
        elif opcode_name == "call":
            return "%s %s" % (opcode_name, I(imm))
        elif opcode_name == "ja":
            return "%s %s" % (opcode_name, O(off))
        elif source == 0:
            return "%s %s, %s, %s" % (opcode_name, R(dst_reg), I(imm), O(off))
        else:
            return "%s %s, %s, %s" % (opcode_name, R(dst_reg), R(src_reg), O(off))
    elif cls == BPF_CLASS_LD or cls == BPF_CLASS_LDX or cls == BPF_CLASS_ST or cls == BPF_CLASS_STX:
        size = (code >> 3) & 3
        mode = (code >> 5) & 7
        mode_name = MODES.get(mode, str(mode))
        # TODO use different syntax for non-MEM instructions
        size_name = SIZES.get(size, str(size))
        if code == 0x18: # lddw
            _, _, _, imm2 = Inst.unpack_from(data, offset+8)
            imm = (imm2 << 32) | imm
            return "%s %s, %s" % (class_name + size_name, R(dst_reg), I(imm))
        elif code == 0x00:
            # Second instruction of lddw
            return None
        elif cls == BPF_CLASS_LDX:
            return "%s %s, %s" % (class_name + size_name, R(dst_reg), M(R(src_reg), off))
        elif cls == BPF_CLASS_ST:
            return "%s %s, %s" % (class_name + size_name, M(R(dst_reg), off), I(imm))
        elif cls == BPF_CLASS_STX:
            return "%s %s, %s" % (class_name + size_name, M(R(dst_reg), off), R(src_reg))
        else:
            return "unknown mem instruction %#x" % code
    else:
        return "unknown instruction %#x" % code

def disassemble(data):
    output = io()
    offset = 0
    while offset < len(data):
        s = disassemble_one(data, offset)
        if s:
            output.write(s + "\n")
        offset += 8
    return output.getvalue()
