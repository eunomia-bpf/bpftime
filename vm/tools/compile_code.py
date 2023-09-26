# encoding: utf8
import os
import sys
from elftools.elf.elffile import ELFFile
import argparse
import subprocess
import ubpf.disassembler
import struct

"""
ebpf code compiling tool, run in linux (or windows) python3.
clang (>= 3.7) is requried.
"""

CODE_TEMPLATE = """
#ifndef EBPF_CODE_H_
#define EBPF_CODE_H_

const unsigned char ebpf_code[] = ""
BYTE_CODE
"";

#endif
"""

CODE_FILE = "ebpf_code.h"
CODEO = "code.o"
OUTPUT = None # "prog.bin"

def save_byte_code(byte_code):
	out_path = CODE_FILE
	# unsigned char str
	uc_str = "".join('\\x{:02x}'.format(c) for c in byte_code)
	# padding = 8 - (len(uc_str) % 8)
	# uc_str += "".join('\\x00' * padding)
	# print("padding", padding, "total", len(uc_str))
	# print(byte_code, uc_str)
	print("code bytes: ", len(byte_code), len(uc_str));
	code_lines = []
	pos, li_sz = 0, 100
	while pos < len(uc_str):
		code_lines.append('"{}"'.format(uc_str[pos:pos+li_sz]))
		pos += li_sz
	fmt_str = "\n".join(code_lines)
	code = CODE_TEMPLATE.replace("BYTE_CODE", fmt_str)
	print("save byte code to", out_path)
	print(fmt_str)
	with open(out_path, "w") as fp:
		fp.write(code)

def bytes_to_str_escape(bys):
	return "".join('\\x{:02x}'.format(c) for c in bys)

def dump_elf_text(prog):
	# 从clang编译的二进制中导出prog
	with open(prog, "rb") as fp:
		elf = ELFFile(fp)
		# for section in elf.iter_sections():
		# 	print(hex(section['sh_addr']), section.name)
		code = elf.get_section_by_name('.text')
		ops = code.data()
		# print("binary code:", bytes_to_str_escape(ops))
		# addr = code['sh_addr']
		# md = Cs(CS_ARCH_X86, CS_MODE_64)
		# for i in md.disasm(ops, addr):        
		# 	print(f'0x{i.address:x}:\t{i.mnemonic}\t{i.op_str}')
		write_binary(ops)

def write_binary(ops):
	save_byte_code(ops)
	if OUTPUT:
		with open(OUTPUT, "wb") as fp:
			fp.write(ops)
	print("disassemble: ")
	data = ubpf.disassembler.disassemble(ops)
	for pc, li in enumerate(data.split("\n")):
		print(pc, li)

def compile_code(src):
	cmd = f"clang -g -emit-llvm -c {src} -o - | llc -march=bpf -filetype=obj -o code.o"
	# cmd = f"clang -O2 -emit-llvm -c {src} -o - | llc -march=bpf -filetype=obj -o code.o"
	# cmd = f"clang -O2 -target bpf -c {src} -o {CODEO} "
	exec_command(cmd, os.getcwd())
	if os.path.exists(CODEO):
		dump_elf_text(CODEO)

def assemble_to_bytecode(asm):
	# only support python2
	# import ubpf.assembler
	# code = ubpf.assembler.assemble(data)
	cmd = "python2 ubpf-assembler.py {} > code.o ".format(asm)
	exec_command(cmd)
	with open("code.o", "rb") as fp:
		code = fp.read()
	write_binary(code)

def exec_command(cmd, cwd=os.getcwd()):
	print(f"Run cmd '{cmd}' in '{cwd}'")
	try:
		result = subprocess.run(cmd, cwd=cwd, shell=True) # capture_output=True
		if result.returncode != 0:
			msg = f"returncode: {result.returncode} cmd: '{result.args}' err:{result.stderr}"
			print("ERROR", msg)
			return False
	except Exception as ex:
		import traceback
		traceback.print_exc()
		return False
	return True

def setup_args():
	parser = argparse.ArgumentParser(prog="compile_ebpf", epilog="e.g. python3 compile_ebpf.py -s code.c")
	parser.add_argument("-s", "--src", metavar="src", help="choose ebpf src.")
	parser.add_argument("-a", "--asm", metavar="asm", help="compile asm.")
	parser.add_argument("-o", "--output", metavar="output", help="set output bin file.")
	parser.add_argument("-f", "--headerfile", metavar="header", help="set output header file.")
	args = parser.parse_args()
	if len(sys.argv) == 1:
	# if not any(vars(args).values()):
		parser.print_help()
		sys.exit(1)
	return args

def main():
	args = setup_args()
	if args.output:
		global OUTPUT
		OUTPUT = args.output
	if args.headerfile:
		global CODE_FILE
		CODE_FILE = args.headerfile
	if args.src:
		compile_code(args.src)
	elif args.asm:
		assemble_to_bytecode(args.asm)
	if os.path.exists(CODEO):
		os.remove(CODEO)

if __name__ == "__main__":
	main()
