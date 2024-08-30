# xdp in bpftime

You can load the xdp program into userspace eBPF runtime, and exec it with DPDK or AF_XDP. This allows:

- `Faster than kernel eBPF`: DPDK can be faster than XDP driver mode, and bpftime can be faster than kernel in some cases.
- Compare to the ubpf in DPDK, this enable:
  - `Control plane application support`: enable the control library to operate the eBPF map, control load and unload the eBPF program dynamically.
  - `More maps and helpers compatible with kernel`: 

## load the XDP into the userspace eBPF runtime

Create a virtual network device for test:

```sh
sudo ip link add veth0 type veth peer name veth1
```

Attach the netdev:

```sh
LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so example/xdp-counter/xdp-counter example/xdp-counter/.output/xdp-counter.bpf.o veth1 example/xdp-counter/base.btf
```

- The `example/xdp-counter/base.btf` is for relocation on userspace xdp. See [runtime/extension/userspace_xdp.h](../../runtime/extension/userspace_xdp.h) for how the userspace `xdp_md` looks like.

## Run XDP program in userspace

See the driver progam in <https://github.com/eunomia-bpf/XDP-eBPF-in-DPDK>

## TODO: support old XDP attach

Currently bpftime only support attach XDP program with `bpf_link`. We need to handle the old attahc approach later.

```sh
newfstatat(1, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
socket(AF_UNIX, SOCK_DGRAM|SOCK_CLOEXEC, 0) = 7
ioctl(7, SIOCGIFINDEX, {ifr_name="veth1", ifr_ifindex=5}) = 0
close(7)                                = 0
socket(AF_NETLINK, SOCK_RAW|SOCK_CLOEXEC, NETLINK_ROUTE) = 7
setsockopt(7, SOL_NETLINK, NETLINK_EXT_ACK, [1], 4) = 0
bind(7, {sa_family=AF_NETLINK, nl_pid=0, nl_groups=00000000}, 12) = 0
getsockname(7, {sa_family=AF_NETLINK, nl_pid=405177, nl_groups=00000000}, [12]) = 0
sendto(7, [{nlmsg_len=52, nlmsg_type=RTM_SETLINK, nlmsg_flags=NLM_F_REQUEST|NLM_F_ACK, nlmsg_seq=1724212008, nlmsg_pid=0}, {ifi_family=AF_UNSPEC, ifi_type=ARPHRD_NETROM, ifi_index=if_nametoindex("veth1"), ifi_flags=0, ifi_change=0}, [{nla_len=20, nla_type=NLA_F_NESTED|IFLA_XDP}, [[{nla_len=8, nla_type=IFLA_XDP_FD}, 6], [{nla_len=8, nla_type=IFLA_XDP_FLAGS}, XDP_FLAGS_UPDATE_IF_NOEXIST]]]], 52, 0, NULL, 0) = 52
recvmsg(7, {msg_name=NULL, msg_namelen=0, msg_iov=[{iov_base=[{nlmsg_len=36, nlmsg_type=NLMSG_ERROR, nlmsg_flags=NLM_F_CAPPED, nlmsg_seq=1724212008, nlmsg_pid=405177}, {error=0, msg={nlmsg_len=52, nlmsg_type=RTM_SETLINK, nlmsg_flags=NLM_F_REQUEST|NLM_F_ACK, nlmsg_seq=1724212008, nlmsg_pid=0}}], iov_len=4096}], msg_iovlen=1, msg_controllen=0, msg_flags=0}, MSG_PEEK|MSG_TRUNC) = 36
recvmsg(7, {msg_name=NULL, msg_namelen=0, msg_iov=[{iov_base=[{nlmsg_len=36, nlmsg_type=NLMSG_ERROR, nlmsg_flags=NLM_F_CAPPED, nlmsg_seq=1724212008, nlmsg_pid=405177}, {error=0, msg={nlmsg_len=52, nlmsg_type=RTM_SETLINK, nlmsg_flags=NLM_F_REQUEST|NLM_F_ACK, nlmsg_seq=1724212008, nlmsg_pid=0}}], iov_len=4096}], msg_iovlen=1, msg_controllen=0, msg_flags=0}, 0) = 36
close(7)                                = 0
```
