#ifndef BPFTIME_EXTENSION_USERSPACE_XDP_HPP
#define BPFTIME_EXTENSION_USERSPACE_XDP_HPP

#include <cstdint>

struct xdp_md_userspace
{
	uint64_t data;
	uint64_t data_end;
	uint32_t data_meta;
	uint32_t ingress_ifindex;
	uint32_t rx_queue_index;
	uint32_t egress_ifindex;
	// additional fields
	uint64_t buffer_start; // record the start of the available buffer
	uint64_t buffer_end; // record the end of the available buffer
};

#endif // BPFTIME_EXTENSION_USERSPACE_XDP_HPP
