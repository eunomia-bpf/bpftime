all: ../../example/libbpf-tools/syscount/syscount index.html

index.html:
	echo "Hello, World!" > index.html

../../example/tracing/syscount/syscount:
	make -C ../../example/tracing/syscount

clean:
	rm -f index.html
	
