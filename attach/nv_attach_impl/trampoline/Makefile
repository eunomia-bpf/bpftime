NVCC ?= nvcc
PTX_TARGET_ARCH ?= sm_75

.PHONY: all clean

all: ../trampoline_ptx.h test

# The trampoline PTX is concatenated with application PTX before being handed to
# ptxas, so nvcc's generic string literal symbols ($str, $str$1, ...) and debug
# line directives must be namespaced/stripped to avoid duplicate definitions.
# `bpf_main` in default_trampoline.cu is only a demo kernel for the standalone
# `test` binary; the real one comes from the compiled eBPF program, so it must be
# dropped here too, otherwise ptxas reports a duplicate definition of bpf_main.
../trampoline_ptx.h: default_trampoline.ptx
	@echo "Generating trampoline_ptx.h..."
	@echo 'static const char TRAMPOLINE_PTX[] = R"(' > $@
	@grep -vE '^[[:space:]]*\.(file|loc)[[:space:]]' $< \
		| awk '!skip && /^[[:space:]]*(\/\/[[:space:]]*\.globl[[:space:]]+bpf_main|(\.visible[[:space:]]+)?\.(entry|func)[[:space:]]+bpf_main)/ { skip = 1 } \
		       skip { if ($$0 == "}") skip = 0; next } { print }' \
		| sed 's/\$$str/\$$__bpftime_tramp_str/g' >> $@
	@echo ')";' >> $@
	@echo "Generated $@"

default_trampoline.ptx: default_trampoline.cu
	$(NVCC) -ptx -O0 -rdc=true -arch=$(PTX_TARGET_ARCH) $< -o $@

# compile it to exec for test
test: default_trampoline.cu
	$(NVCC) -lineinfo -O2 -arch=$(PTX_TARGET_ARCH) default_trampoline.cu -o test

clean:
	rm -f default_trampoline.ptx
	rm -f ../trampoline_ptx.h 
	rm -f test
