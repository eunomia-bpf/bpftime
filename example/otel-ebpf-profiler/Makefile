# SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
CC ?= gcc

.PHONY: all upstream-otel otel-full-run benchmark clean

all: victim

upstream-otel:
	scripts/build-upstream-otel.sh

otel-full-run: victim
	scripts/run-full-otel-profiler.sh

benchmark: victim
	scripts/benchmark.sh

victim: victim-src/victim.c
	$(CC) -g -O2 -Wall $< -o $@

clean:
	rm -f victim
