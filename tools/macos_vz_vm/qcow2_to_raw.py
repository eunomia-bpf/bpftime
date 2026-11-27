#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Minimal QCOW2 v3 -> raw converter for bpftime-vz.
#
# Virtualization.framework boots only raw disk images, while Ubuntu ships
# its arm64 cloud image as QCOW2 v3 and stock macOS has no qemu-img and there
# is no official raw arm64 cloud image. This decodes exactly the cluster
# kinds Ubuntu uses: plain, zero/unallocated (written as sparse holes) and
# zlib raw-deflate compressed clusters (the v3 compressed descriptor).
#
# Usage: qcow2_to_raw.py input.qcow2 output.raw [target_size_bytes]
import struct
import sys
import zlib

L2_OFFSET_MASK = 0x00FFFFFFFFFFFFFF
QCOW_OFLAG_ZERO = 1 << 0
QCOW_OFLAG_COMPRESSED = 1 << 62


def die(msg):
    sys.exit(f"qcow2_to_raw: {msg}")


def convert(src_path, dst_path, target_size):
    with open(src_path, "rb") as src:
        header = src.read(112)
        if len(header) < 112 or header[0:4] != b"QFI\xfb":
            die("input is not a QCOW2 image")
        (_, version, backing_off, _backing_size, cluster_bits, virtual_size,
         crypt_method, l1_size, l1_off, _rc_off, _rc_clusters, _nb_snap,
         _snap_off, incompat, _compat, _autoclear, _refcount_order,
         _header_len) = \
            struct.unpack_from(">4sIQIIQIIQQIIQQQQII", header)
        if version < 2 or crypt_method != 0 or (incompat & ~0x1):
            die("unsupported qcow2 version, encryption or feature bits")
        # Overlays inherit unallocated clusters from a backing image, which
        # this single-image converter cannot resolve; reject rather than emit
        # sparse zeroes for data that should come from the backing chain.
        if backing_off != 0:
            die("backing-file overlays are not supported")
        if not 9 <= cluster_bits <= 30:
            die("invalid cluster_bits")
        cluster_size = 1 << cluster_bits
        l2_entries = cluster_size // 8
        out_size = target_size or virtual_size
        if out_size < virtual_size:
            die("target size smaller than virtual disk size")
        clusters_total = (virtual_size + cluster_size - 1) // cluster_size
        required_l1 = (clusters_total + l2_entries - 1) // l2_entries
        if l1_off == 0 or l1_size < required_l1:
            die("L1 table does not cover the virtual disk")
        src.seek(l1_off)
        l1_table = struct.unpack(f">{l1_size}Q", src.read(l1_size * 8))
        done = 0
        with open(dst_path, "wb") as dst:
            dst.truncate(out_size)
            for l1_entry in l1_table:
                if l1_entry == 0:
                    done += min(l2_entries, clusters_total - done)
                    continue
                src.seek(l1_entry & L2_OFFSET_MASK)
                l2_table = struct.unpack(
                    f">{l2_entries}Q", src.read(cluster_size))
                for l2_entry in l2_table:
                    if done >= clusters_total:
                        break
                    out_off = done * cluster_size
                    done += 1
                    # QEMU v3 compressed descriptor: the host byte offset
                    # occupies the low (62 - width) bits and the count of
                    # additional 512-byte sectors the next 'width' bits just
                    # under the compressed flag (bit 62). This keeps the
                    # offset wide enough for images beyond 4 GiB.
                    if l2_entry & QCOW_OFLAG_COMPRESSED:
                        width = cluster_bits - 8
                        comp_shift = 62 - width
                        comp_off = l2_entry & ((1 << comp_shift) - 1)
                        sectors = ((l2_entry >> comp_shift)
                                   & ((1 << width) - 1)) + 1
                        comp_len = sectors * 512
                        src.seek(comp_off)
                        blob = src.read(comp_len)
                        if len(blob) != comp_len:
                            die(f"truncated compressed cluster {done - 1}")
                        data = zlib.decompressobj(-15).decompress(blob)
                        if len(data) != cluster_size:
                            die(f"cluster {done - 1}: inflated "
                                f"{len(data)} bytes, expected {cluster_size}")
                    elif l2_entry == 0 or (l2_entry & QCOW_OFLAG_ZERO):
                        continue  # sparse hole already reads back as zeros
                    else:
                        src.seek(l2_entry & L2_OFFSET_MASK)
                        data = src.read(cluster_size)
                        if len(data) != cluster_size:
                            die(f"truncated image: short cluster {done - 1}")
                    # A non-cluster-aligned virtual disk ends in a partial
                    # cluster; never write past the requested output size.
                    available = out_size - out_off
                    if available > 0:
                        dst.seek(out_off)
                        dst.write(data[:available])


def main():
    if len(sys.argv) < 3:
        die("usage: qcow2_to_raw.py input.qcow2 output.raw [target_bytes]")
    target = int(sys.argv[3]) if len(sys.argv) > 3 else None
    convert(sys.argv[1], sys.argv[2], target)


if __name__ == "__main__":
    main()
