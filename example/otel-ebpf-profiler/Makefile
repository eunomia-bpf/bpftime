# SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
CC ?= cc
CFLAGS ?= -O2 -g -Wall -Wextra

.PHONY: all clean

all: victim

victim: victim-src/victim.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f victim
