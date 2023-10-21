#ifndef BPFTIME_JSON_EXPORTER_HPP
#define BPFTIME_JSON_EXPORTER_HPP

#include <string>
#include <sstream>
#include <stdio.h>
#include <iomanip>

namespace bpftime
{
static inline std::string bufferToHexString(const unsigned char *buffer,
					    size_t bufferSize)
{
	std::stringstream ss;
	ss << std::hex << std::setfill('0');
	for (size_t i = 0; i < bufferSize; i++) {
		ss << std::setw(2) << static_cast<int>(buffer[i]);
	}
	return ss.str();
}

static inline int hexStringToBuffer(const std::string &hexString,
				     unsigned char *buffer, size_t bufferSize)
{
	if (hexString.length() != bufferSize * 2) {
		return -1;
	}

	for (size_t i = 0; i < bufferSize; i++) {
		std::string byteString = hexString.substr(i * 2, 2);
		buffer[i] = static_cast<unsigned char>(
			std::stoi(byteString, nullptr, 16));
	}
    return 0;
}

void bpftime_import_global_shm_from_json(const char *filename);
void bpftime_export_global_shm_to_json(const char *filename);

} // namespace bpftime
#endif // BPFTIME_JSON_EXPORTER_HPP
