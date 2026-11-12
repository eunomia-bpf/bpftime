CC = gcc
CFLAGS = -Wall -Wextra -fpic -O2
LDFLAGS = -shared

all: libfilter_impl.so

libfilter_impl.so: filter_impl.o
	$(CC) $(LDFLAGS) -o $@ $<

filter_impl.o: filter_impl.c filter_impl.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o *.so

.PHONY: all clean 