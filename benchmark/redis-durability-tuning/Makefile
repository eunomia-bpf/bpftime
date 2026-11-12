CC=gcc
CFLAGS=-Wall

.PHONY: all build clean redis batch_process delayed-fsync fsync-fast-notify poc-iouring-minimal

# Default target builds all implementations
all: build redis

# Build all BPF implementations
build: batch_process delayed-fsync fsync-fast-notify poc-iouring-minimal

# Individual implementation targets
batch_process:
	@echo "Building batch_process implementation..."
	make -C batch_process

delayed-fsync:
	@echo "Building delayed-fsync implementation..."
	make -C delayed-fsync

fsync-fast-notify:
	@echo "Building fsync-fast-notify implementation..."
	make -C fsync-fast-notify

poc-iouring-minimal:
	@echo "Building poc-iouring-minimal implementation..."
	make -C poc-iouring-minimal

# Build Redis
redis:
	@echo "Cloning and building Redis..."
	git clone https://github.com/eunomia-bpf/redis --depth 1 || (cd redis && git pull)
	cd redis && make

# Clean all builds
clean:
	@echo "Cleaning all implementations..."
	make -C batch_process clean || true
	make -C delayed-fsync clean || true
	make -C fsync-fast-notify clean || true
	make -C poc-iouring-minimal clean || true
	@echo "Note: Redis build is not cleaned. To clean Redis, run 'make clean' in the redis directory."

# Print help information
help:
	@echo "Redis Durability Tuning Benchmark Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  all              - Build all implementations and Redis (default)"
	@echo "  build            - Build all BPF implementations"
	@echo "  redis            - Clone and build Redis"
	@echo "  batch_process    - Build batch_process implementation"
	@echo "  delayed-fsync    - Build delayed-fsync implementation" 
	@echo "  fsync-fast-notify - Build fsync-fast-notify implementation"
	@echo "  poc-iouring-minimal - Build poc-iouring-minimal implementation"
	@echo "  clean            - Clean all BPF implementations"
	@echo "  help             - Show this help message"
