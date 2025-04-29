all: ../../example/libbpf-tools/syscount/syscount index.html

index.html:
	echo "Hello, World!" > index.html

../../example/libbpf-tools/syscount/syscount:
	make -C ../../example/libbpf-tools/syscount

clean:
	rm -f index.html
	
