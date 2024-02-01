#!/usr/bin/env python
"""
Expand testcase into individual files
"""
import os
import sys
import struct
import testdata
import argparse

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
if os.path.exists(os.path.join(ROOT_DIR, "ubpf")):
    # Running from source tree
    sys.path.insert(0, ROOT_DIR)

import ubpf.assembler
import ubpf.disassembler

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('name')
    parser.add_argument('path')
    args = parser.parse_args()

    data = testdata.read(_test_data_dir, args.name + '.data')
    assert data

    if not os.path.isdir(args.path):
        os.makedirs(args.path)

    def writefile(name, contents):
        open("%s/%s" % (args.path, name), "wb").write(contents)

    if 'mem' in data:
        writefile('mem', data['mem'])

        # Probably a packet, so write out a pcap file
        writefile('pcap',
            struct.pack('=IHHIIIIIIII',
                0xa1b2c3d4, # magic
                2, 4, # version
                0, # time zone offset
                0, # time stamp accuracy
                65535, # snapshot length
                1, # link layer type
                0, 0, # timestamp
                len(data['mem']), # length
                len(data['mem'])) # length
            + data['mem'])

    if 'raw' in data:
        code = b''.join(struct.pack("=Q", x) for x in data['raw'])
    elif 'asm' in data:
        code = ubpf.assembler.assemble(data['asm'])
    else:
        code = None

    if code:
        writefile('code', code)

    if 'asm' in data:
        writefile('asm', data['asm'].encode())
    elif code:
        writefile('asm', ubpf.disassembler.disassemble(code))

    if 'pyelf' in data:
        from test_elf import generate_elf
        elf = generate_elf(data['pyelf'])
        writefile('elf', elf)

if __name__ == "__main__":
    main()
