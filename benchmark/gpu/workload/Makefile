# SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)

.PHONY: all
all: vec_add

vec_add: vec_add.cu
	@if command -v nvcc >/dev/null 2>&1; then \
		nvcc -cudart shared vec_add.cu -o vec_add -g; \
	else \
		echo "Warning: CUDA not found, skipping vec_add build"; \
	fi

.PHONY: clean
clean:
	rm -f vec_add
