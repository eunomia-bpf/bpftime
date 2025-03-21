CXX = clang
CFLAGS = -nostdinc \
         -I../../third_party/macos \
         -I../../runtime/include \
         -I../../vm/compat/include \
         -I../../third_party/spdlog/include \
         -I../../vm/vm-core/include \
         -D__KERNEL__ \
         -D__BPF_TRACING__ \
         -D__TARGET_ARCH_bpf \
         -DCONFIG_64BIT

SRC = mapop.bpf.c
TGT = main

main: $(SRC)
	$(CXX) -g $(CFLAGS) -target bpf -Wall -O2 -c $(SRC) -o $(TGT)

.PHONY: clean

clean:
	rm -rf $(TGT)