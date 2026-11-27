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
        (_, version, _, _, cluster_bits, virtual_size, crypt_method,
         l1_size, l1_off, _, _, _, _, incompat, _, _, _, _) = \
            struct.unpack_from(">4sIQIIQIIQQIIQQQQII", header)
        if version < 2 or crypt_method != 0 or (incompat & ~0x1):
            die("unsupported qcow2 version, encryption or feature bits")
        cluster_size = 1 << cluster_bits
        l2_entries = cluster_size // 8
        out_size = target_size or virtual_size
        if out_size < virtual_size:
            die("target size smaller than virtual disk size")
        src.seek(l1_off)
        l1_table = struct.unpack(f">{l1_size}Q", src.read(l1_size * 8))
        clusters_total = (virtual_size + cluster_size - 1) // cluster_size
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
                    # Compressed takes precedence: its low bits pack the
                    # compressed length, so the zero flag is meaningless there.
                    if l2_entry & QCOW_OFLAG_COMPRESSED:
                        src.seek(l2_entry & 0xFFFFFFFF)
                        data = zlib.decompressobj(-15).decompress(
                            src.read(cluster_size * 2))
                        if len(data) != cluster_size:
                            die(f"cluster {done - 1}: inflated "
                                f"{len(data)} bytes, expected {cluster_size}")
                    elif l2_entry == 0 or (l2_entry & QCOW_OFLAG_ZERO):
                        continue  # sparse hole already reads back as zeros
                    else:
                        src.seek(l2_entry & L2_OFFSET_MASK)
                        data = src.read(cluster_size)
                    dst.seek(out_off)
                    dst.write(data)


def main():
    if len(sys.argv) < 3:
        die("usage: qcow2_to_raw.py input.qcow2 output.raw [target_bytes]")
    target = int(sys.argv[3]) if len(sys.argv) > 3 else None
    convert(sys.argv[1], sys.argv[2], target)


if __name__ == "__main__":
    main()
