#! /bin/sh

# script to print all the syscalls in a system
# usage <script_name> <file_name>
# usage ./generate_syscall_id_table syscalls.txt
if [ "$(uname)" = "Linux" ]; then
    echo "static const char* table=R\"(`
    echo -e '#include <sys/syscall.h>' | \
    cpp -dM | grep '#define __NR_.*[0-9]$' | \
    cut -d' ' -f 2,3 | cut -d_ -f 4-
    `)\";" > $1
elif [ "$(uname)" = "Darwin" ]; then
    echo "static const char* table=R\"(`
    echo '#include <sys/syscall.h>' | \
    gcc -E -dM - | grep '#define SYS_.*[0-9]$' | \
    sed 's/#define SYS_//' | \
    awk '{print $1, $2}'
    `)\";" > $1
fi
