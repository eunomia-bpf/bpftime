#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

::std::unique_ptr<::std::string> patch_ptx(char const *ptx) noexcept;

::std::unique_ptr<::std::vector<::std::uint8_t>> patch_fatbin(::std::uint8_t const *fatbin) noexcept;

::std::unique_ptr<::std::vector<::std::uint8_t>> patch_raw_image(::std::uint8_t const *image) noexcept;
