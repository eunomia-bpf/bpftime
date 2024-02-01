PROG := tcp-sack

$(PROG).asm: $(PROG).bin
	../../bin/ubpf-disassembler $^ $@

$(PROG).bin: $(PROG).o
	objcopy -I elf64-little -O binary $^ $@

$(PROG).o: $(PROG).c
	clang-3.7 -O2 -target bpf -c $^ -o $@
