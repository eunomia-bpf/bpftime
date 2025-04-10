#include <cstdint>
#include <memory>
#include <string>
#include <vector>

extern "C" {
::std::string *cxxbridge1$patch_ptx(char const *ptx) noexcept;

::std::vector<::std::uint8_t> *cxxbridge1$patch_fatbin(::std::uint8_t const *fatbin) noexcept;

::std::vector<::std::uint8_t> *cxxbridge1$patch_raw_image(::std::uint8_t const *image) noexcept;
} // extern "C"

::std::unique_ptr<::std::string> patch_ptx(char const *ptx) noexcept {
  return ::std::unique_ptr<::std::string>(cxxbridge1$patch_ptx(ptx));
}

::std::unique_ptr<::std::vector<::std::uint8_t>> patch_fatbin(::std::uint8_t const *fatbin) noexcept {
  return ::std::unique_ptr<::std::vector<::std::uint8_t>>(cxxbridge1$patch_fatbin(fatbin));
}

::std::unique_ptr<::std::vector<::std::uint8_t>> patch_raw_image(::std::uint8_t const *image) noexcept {
  return ::std::unique_ptr<::std::vector<::std::uint8_t>>(cxxbridge1$patch_raw_image(image));
}
