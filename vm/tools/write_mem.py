#!encoding: utf-8

import struct
import ctypes

MEM = "mem"

class Data(ctypes.Structure):
	_fields_ = [
		('a', ctypes.c_int),
		('b', ctypes.c_int),
		# ('patch_point', ctypes.c_int), # addr or patch id
		# ('code', ctypes) # ebpf bytecode
	]

def write_mem1():
	a, b = 4000, 2
	data = struct.pack('i', a)
	data += struct.pack('i', b)
	print(len(data), data)
	with open(MEM, "wb") as fp:
		fp.write(data)
	d = Data(a, b)
	print(bytearray(d))

def write_mem2():
	with open(MEM, "wb") as fp:
		fp.write(b'123456789   abcdef')
		fp.write(bytes(0x0))

def main():
	write_mem1()
	# write_mem2()


if __name__ == "__main__":
	main()