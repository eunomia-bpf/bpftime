echo "static const char* table=R\"(`
echo -e '#include <sys/syscall.h>' | \
cpp -dM | grep '#define __NR_.*[0-9]$' | \
cut -d' ' -f 2,3 | cut -d_ -f 4-
`)\";" > $1
