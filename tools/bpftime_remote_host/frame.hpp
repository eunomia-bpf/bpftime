/* SPDX-License-Identifier: MIT
 *
 * Wire framing shared by bpftime-remote-host and its control-plane clients.
 * Every message is an 8-byte little-endian header followed by a UTF-8 JSON
 * document and an optional raw binary blob:
 *
 *   struct frame_hdr { uint32_t json_len; uint32_t blob_len; };
 */
#ifndef BPFTIME_REMOTE_FRAME_HPP
#define BPFTIME_REMOTE_FRAME_HPP

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace bpftime
{
namespace remote
{

struct Frame {
	std::string json;
	std::vector<uint8_t> blob;
};

inline void append_u32_le(std::vector<uint8_t> &out, uint32_t v)
{
	out.push_back((uint8_t)(v & 0xff));
	out.push_back((uint8_t)((v >> 8) & 0xff));
	out.push_back((uint8_t)((v >> 16) & 0xff));
	out.push_back((uint8_t)((v >> 24) & 0xff));
}

inline uint32_t read_u32_le(const uint8_t *p)
{
	return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
	       ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

inline std::vector<uint8_t> encode(const std::string &json,
				   const void *blob = nullptr,
				   size_t blob_len = 0)
{
	if (json.size() > UINT32_MAX || blob_len > UINT32_MAX)
		throw std::overflow_error("remote frame too large");
	std::vector<uint8_t> out;
	out.reserve(8 + json.size() + blob_len);
	append_u32_le(out, (uint32_t)json.size());
	append_u32_le(out, (uint32_t)blob_len);
	out.insert(out.end(), json.begin(), json.end());
	if (blob_len > 0) {
		const auto *p = static_cast<const uint8_t *>(blob);
		out.insert(out.end(), p, p + blob_len);
	}
	return out;
}

inline void read_full(int fd, void *buf, size_t n)
{
	auto *p = static_cast<uint8_t *>(buf);
	while (n > 0) {
		ssize_t r = ::read(fd, p, n);
		if (r == 0)
			throw std::runtime_error("peer closed connection");
		if (r < 0) {
			if (errno == EINTR)
				continue;
			throw std::runtime_error(std::string("read failed: ") +
						 strerror(errno));
		}
		p += r;
		n -= (size_t)r;
	}
}

inline Frame recv_frame(int fd)
{
	uint8_t hdr[8];
	read_full(fd, hdr, sizeof(hdr));
	uint32_t json_len = read_u32_le(hdr);
	uint32_t blob_len = read_u32_le(hdr + 4);
	Frame frame;
	frame.json.resize(json_len);
	if (json_len > 0)
		read_full(fd, frame.json.data(), json_len);
	frame.blob.resize(blob_len);
	if (blob_len > 0)
		read_full(fd, frame.blob.data(), blob_len);
	return frame;
}

inline void write_full(int fd, const void *buf, size_t n)
{
	const auto *p = static_cast<const uint8_t *>(buf);
	while (n > 0) {
		ssize_t w = ::write(fd, p, n);
		if (w < 0) {
			if (errno == EINTR)
				continue;
			throw std::runtime_error(std::string("write failed: ") +
						 strerror(errno));
		}
		p += w;
		n -= (size_t)w;
	}
}

inline void send_frame(int fd, const std::string &json,
		       const void *blob = nullptr, size_t blob_len = 0)
{
	auto bytes = encode(json, blob, blob_len);
	write_full(fd, bytes.data(), bytes.size());
}

} // namespace remote
} // namespace bpftime

#endif
