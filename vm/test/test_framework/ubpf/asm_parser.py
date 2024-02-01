#!/usr/bin/env python
from __future__ import print_function
from parcon import *
from collections import namedtuple

hexchars = '0123456789abcdefABCDEF'

Reg = namedtuple("Reg", ["num"])
Imm = namedtuple("Imm", ["value"])
MemRef = namedtuple("MemRef", ["reg", "offset"])

def keywords(vs):
    return First(*[Keyword(SignificantLiteral(v)) for v in vs])

hexnum = SignificantLiteral('0x') + +CharIn(hexchars)
decnum = +Digit()
offset = (CharIn("+-") + Exact(hexnum | decnum))[flatten]["".join][lambda x: int(x, 0)]
imm = (-CharIn("+-") + Exact(hexnum | decnum))[flatten]["".join][lambda x: int(x, 0)][Imm]

reg = Literal('r') + integer[int][Reg]
memref = (Literal('[') + reg + Optional(offset, 0) + Literal(']'))[lambda x: MemRef(*x)]

unary_alu_ops = ['neg', 'neg32', 'le16', 'le32', 'le64', 'be16', 'be32', 'be64']
binary_alu_ops = ['add', 'sub', 'mul', 'div', 'or', 'and', 'lsh', 'rsh',
                  'mod', 'xor', 'mov', 'arsh']
binary_alu_ops.extend([x + '32' for x in binary_alu_ops])

alu_instruction = \
    (keywords(unary_alu_ops) + reg) | \
    (keywords(binary_alu_ops) + reg + "," + (reg | imm))

mem_sizes = ['w', 'h', 'b', 'dw']
mem_store_reg_ops = ['stx' + s for s in mem_sizes]
mem_store_imm_ops = ['st' + s for s in mem_sizes]
mem_load_ops = ['ldx' + s for s in mem_sizes]

mem_instruction = \
    (keywords(mem_store_reg_ops) + memref + "," + reg) | \
    (keywords(mem_store_imm_ops) + memref + "," + imm) | \
    (keywords(mem_load_ops) + reg + "," + memref) | \
    (keywords(["lddw"]) + reg + "," + imm)

jmp_cmp_ops = ['jeq', 'jgt', 'jge', 'jlt', 'jle', 'jset', 'jne', 'jsgt', 'jsge', 'jslt', 'jsle']
jmp_instruction = \
    (keywords(jmp_cmp_ops) + reg + "," + (reg | imm) + "," + offset) | \
    (keywords(['ja']) + offset) | \
    (keywords(['call']) + imm) | \
    (keywords(['exit'])[lambda x: (x, )])

instruction = alu_instruction | mem_instruction | jmp_instruction

start = ZeroOrMore(instruction + Optional(Literal(';'))) + End()

def parse(source):
    return start.parse_string(source)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Assembly parser", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('file', type=argparse.FileType('r'), default='-')
    args = parser.parse_args()
    result = parse(args.file.read())
    for inst in result:
        print(repr(inst))
