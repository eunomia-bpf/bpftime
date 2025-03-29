#include "to_cxx.h"

std::unique_ptr<std::vector<uint8_t>> to_cxx_vec(rust::Slice<const uint8_t> vec) {
    return std::make_unique<std::vector<uint8_t>>(
        vec.data(), vec.data() + vec.size()
    );
}

std::unique_ptr<std::string> to_cxx_string(rust::Str s) {
    return std::make_unique<std::string>(s.data(), s.size());
}
