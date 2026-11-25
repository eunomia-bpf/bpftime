# Patch usdt.bpf.h to allow LINUX_HAS_BPF_COOKIE pre-definition.
# See https://github.com/eunomia-bpf/bpftime/issues/710
file(READ "${INPUT_FILE}" CONTENT)
string(REPLACE
    "extern const _Bool LINUX_HAS_BPF_COOKIE __kconfig;"
    "#ifndef LINUX_HAS_BPF_COOKIE\nextern const _Bool LINUX_HAS_BPF_COOKIE __kconfig;\n#endif"
    PATCHED "${CONTENT}")
file(WRITE "${INPUT_FILE}" "${PATCHED}")
message(STATUS "Patched ${INPUT_FILE}: LINUX_HAS_BPF_COOKIE can now be pre-defined")
