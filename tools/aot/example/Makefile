malloc: main.c
	clang -Xlinker --export-dynamic -O2 -flto main.c do_count.o -o malloc

.PHONY: clean
clean:
	rm -f malloc
