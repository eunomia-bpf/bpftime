file(READ "${INPUT}" bytes HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${bytes}")
file(WRITE "${OUTPUT}" "// Generated from device_helpers.hip.\nstatic constexpr unsigned char rocm_device_helpers[] = {${bytes}};\n")
