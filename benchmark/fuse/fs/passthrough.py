import os
import errno
import signal
import threading
import time
from fuse import FUSE, Operations


class Passthrough(Operations):
    op_count = {}

    def __init__(self, root):
        self.root = root

    def _increment_op_count(self, op_name):
        if op_name in self.op_count:
            self.op_count[op_name] += 1
        else:
            self.op_count[op_name] = 1

    def _full_path(self, partial):
        self._increment_op_count("_full_path")
        if partial.startswith("/"):
            partial = partial[1:]
        path = os.path.join(self.root, partial)
        return path

    def getattr(self, path, fh=None):
        self._increment_op_count("getattr")
        full_path = self._full_path(path)
        if full_path.startswith("/home/yunwei/linux/arch"):
            # throw error
            raise OSError(errno.EACCES, "")
        st = os.lstat(full_path)
        return dict(
            (key, getattr(st, key))
            for key in (
                "st_atime",
                "st_ctime",
                "st_gid",
                "st_mode",
                "st_mtime",
                "st_nlink",
                "st_size",
                "st_uid",
            )
        )

    def readdir(self, path, fh):
        self._increment_op_count("readdir")
        full_path = self._full_path(path)
        # print(full_path)
        # block access to arch directory
        if full_path.startswith("/home/yunwei/linux/arch"):
            # return error
            raise OSError(errno.EACCES, "")
        dirents = [".", ".."]
        if os.path.isdir(full_path):
            dirents.extend(os.listdir(full_path))
        for r in dirents:
            yield r

    def open(self, path, flags):
        self._increment_op_count("open")
        full_path = self._full_path(path)
        # print(full_path)
        # block access to arch directory
        if full_path.startswith("/home/yunwei/linux/arch"):
            # throw error
            raise OSError(errno.EACCES, "")
        return os.open(full_path, flags)

    def create(self, path, mode, fi=None):
        self._increment_op_count("create")
        full_path = self._full_path(path)
        return os.open(full_path, os.O_WRONLY | os.O_CREAT, mode)

    def read(self, path, length, offset, fh):
        self._increment_op_count("read")
        os.lseek(fh, offset, os.SEEK_SET)
        return os.read(fh, length)

    def write(self, path, buf, offset, fh):
        self._increment_op_count("write")
        os.lseek(fh, offset, os.SEEK_SET)
        return os.write(fh, buf)

    def truncate(self, path, length, fh=None):
        self._increment_op_count("truncate")
        full_path = self._full_path(path)
        with open(full_path, "r+") as f:
            f.truncate(length)

    def flush(self, path, fh):
        self._increment_op_count("flush")
        return os.fsync(fh)

    def release(self, path, fh):
        self._increment_op_count("release")
        return os.close(fh)

    def fsync(self, path, fdatasync, fh):
        self._increment_op_count("fsync")
        return self.flush(path, fh)


def print_op_counts():
    global fs
    while True:
        print("\nOperation counts:")
        for op, count in fs.op_count.items():
            print(f"{op}: {count}")
        time.sleep(10)  # 等待10秒


def main(mountpoint, root):
    global fs
    fs = Passthrough(root)

    # 启动后台线程进行日志记录
    log_thread = threading.Thread(target=print_op_counts, daemon=True)
    log_thread.start()

    FUSE(fs, mountpoint, nothreads=True, foreground=True, allow_root=True)

if __name__ == "__main__":
    import sys

    main(sys.argv[2], sys.argv[1])
