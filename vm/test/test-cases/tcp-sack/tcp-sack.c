#include <stdint.h>
#include <net/ethernet.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>

struct tcp_option
{
    uint8_t kind;
    uint8_t length;
};

uint64_t
entry(void* pkt)
{
    struct ether_header* ether_header = (void*)pkt;

    if (ether_header->ether_type != __builtin_bswap16(0x0800)) {
        return 0;
    }

    struct iphdr* iphdr = (void*)(ether_header + 1);
    if (iphdr->protocol != 6) {
        return 0;
    }

    struct tcphdr* tcphdr = (void*)iphdr + iphdr->ihl * 4;

    void* options_start = (void*)(tcphdr + 1);
    int options_length = tcphdr->doff * 4 - sizeof(*tcphdr);
    int offset = 0;

    while (offset < options_length) {
        struct tcp_option* option = options_start + offset;
        if (option->kind == 0) {
            /* End of options */
            break;
        } else if (option->kind == 1) {
            /* NOP */
            offset++;
        } else if (option->kind == 5) {
            /* SACK */
            return 1;
        }

        offset += option->length;
    }

    return 0;
}
