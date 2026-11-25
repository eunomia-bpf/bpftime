#! /bin/sh

# script to print all the syscalls in a system
# usage <script_name> <file_name>
# usage ./generate_syscall_id_table syscalls.txt

generate_linux_table() {
    printf '#include <sys/syscall.h>\n' | cpp -dM | \
    awk '$1 == "#define" && $2 ~ /^__NR_[A-Za-z0-9_]+$/ && $2 != "__NR_syscalls" && $2 != "__NR_arch_specific_syscall" {
        printf "\"%s\" %s\n", substr($2, 6), $2
    }' | { printf '#include <sys/syscall.h>\n'; cat; } | cpp -P | \
    awk '
        /^"[^"]+"[ \t]+[0-9()+ \-]+$/ {
            name = $1
            gsub(/^"|"$/, "", name)
            sub(/^"[^"]+"[ \t]+/, "")
            expr = $0
            gsub(/[() \t]/, "", expr)
            if (expr ~ /^[0-9]+$/) {
                print name, expr
            } else if (expr ~ /^[0-9]+\+[0-9]+$/) {
                split(expr, parts, "+")
                print name, parts[1] + parts[2]
            }
        }
    '
}

if [ "$(uname)" = "Linux" ]; then
    {
        printf 'static const char* table=R"(\n'
        generate_linux_table
        printf ')";\n'
    } > "$1"
elif [ "$(uname)" = "Darwin" ]; then
    echo "static const char* table=R\"(`
    echo '#include <sys/syscall.h>' | \
    gcc -E -dM - | grep '#define SYS_.*[0-9]$' | \
    sed 's/#define SYS_//' | \
    awk '{print $1, $2}'
    `)\";" > "$1"
fi
