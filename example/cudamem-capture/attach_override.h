/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */

#ifndef D951AEE1_B02B_47EF_A451_A69982F0386C
#define D951AEE1_B02B_47EF_A451_A69982F0386C
#ifndef BPFTIME_UREPLACE_ATTACH_H
#define BPFTIME_UREPLACE_ATTACH_H

#include <unistd.h>
#include <stdlib.h>
#include <syscall.h>
#include <linux/perf_event.h>
#include <linux/bpf.h>
#include <bpf/bpf.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <ctype.h>

long elf_find_func_offset_from_file(const char *binary_path, const char *name);

#define PERF_UPROBE_REF_CTR_OFFSET_BITS 32
#define PERF_UPROBE_REF_CTR_OFFSET_SHIFT 32
#define BPF_TYPE_UPROBE_OVERRIDE 1008


/* 仅保存读取 PTX 所需的最小信息 */
struct ptx_fd {
	int          fd;     /* 原始文件描述符           */
	const char  *ptx;    /* 映射到用户态的首地址       */
	size_t       size;   /* 文件长度（字节）           */
};

/* 关闭 / 解除映射；允许传入 NULL */
static inline void ptx_close(struct ptx_fd *p)
{
	if (!p)
		return;

	if (p->ptx && p->ptx != MAP_FAILED)
		munmap((void *)p->ptx, p->size);

	if (p->fd >= 0)
		close(p->fd);

	p->fd  = -1;
	p->ptx = NULL;
	p->size = 0;
}

/* 判断 name 是否以 "__cuda" 结尾 */
static inline int is_cuda_symbol(const char *name)
{
	static const char suffix[] = "__cuda";
	size_t nlen = strlen(name);
	size_t slen = sizeof(suffix) - 1;   /* 不含 '\0' */

	return nlen >= slen && memcmp(name + nlen - slen, suffix, slen) == 0;
}

/*
 * 只支持常规 PTX 文本文件：
 *   - path 指向 .ptx            （若是 fatbin 请在调用前先展开）
 *   - out  成功时被填充
 * 返回 0 成功；否则为 -errno
 */
static inline long ptx_open(const char *path, struct ptx_fd *out)
{
	struct stat st;
	int         fd;
	void       *map;

	if (!path || !out)
		return -EINVAL;

	fd = open(path, O_RDONLY);
	if (fd < 0)
		return -errno;

	if (fstat(fd, &st)) {
		int err = errno;
		close(fd);
		return -err;
	}

	if (!S_ISREG(st.st_mode) || st.st_size == 0) {
		close(fd);
		return -EINVAL;
	}

	map = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (map == MAP_FAILED) {
		int err = errno;
		close(fd);
		return -err;
	}

	/* 填充结构体供后续使用 */
	out->fd   = fd;
	out->ptx  = (const char *)map;
	out->size = (size_t)st.st_size;
	return 0;
}

static inline __u64 ptr_to_u64(const void *ptr)
{
	return (__u64)(unsigned long)ptr;
}

static int perf_event_open_with_override(const char *name, uint64_t offset,
					 int pid, size_t ref_ctr_off, int type)
{
	const size_t attr_sz = sizeof(struct perf_event_attr);
	struct perf_event_attr attr;
	int pfd;

	if ((__u64)ref_ctr_off >= (1ULL << PERF_UPROBE_REF_CTR_OFFSET_BITS))
		return -EINVAL;

	memset(&attr, 0, attr_sz);

	attr.size = attr_sz;
	attr.type = type;
	attr.config |= (__u64)ref_ctr_off << PERF_UPROBE_REF_CTR_OFFSET_SHIFT;
	attr.config1 = ptr_to_u64(name); /* kprobe_func or uprobe_path */
	attr.config2 = offset; /* kprobe_addr or probe_offset */

	/* pid filter is meaningful only for uprobes */
	pfd = syscall(__NR_perf_event_open, &attr, pid < 0 ? -1 : pid /* pid */,
		      pid == -1 ? 0 : -1 /* cpu */, -1 /* group_fd */,
		      PERF_FLAG_FD_CLOEXEC);
	return pfd >= 0 ? pfd : -errno;
}

static inline long ptx_find_func_offset(const char *ptx,
					const char *name)
{
	if (!ptx || !name)
		return -EINVAL;

	/* ------- 1) 去掉 "__cuda" 后缀 ------- */
	const char  *base = name;
	char         buf[256];                    /* 临时缓冲 */
	const char   suffix[] = "__cuda";
	size_t       nlen  = strlen(name);
	size_t       slen  = sizeof(suffix) - 1;

	if (nlen >= slen && memcmp(name + nlen - slen, suffix, slen) == 0) {
		size_t blen = nlen - slen;        /* 剥掉后缀 */
		if (blen >= sizeof(buf))
			return -ENAMETOOLONG;
		memcpy(buf, name, blen);
		buf[blen] = '\0';
		base = buf;
	}

	size_t       blen = strlen(base);
	const char  *p    = ptx;

	/* 辅助宏：判断字符是否属于 C 标识符 [a‑zA‑Z0‑9_] */
#define IS_IDCHAR(c)  (isalnum((unsigned char)(c)) || (c) == '_')

	/* ------- 2) 逐次 strstr() 搜索 ------- */
	while ((p = strstr(p, base)) != NULL) {
		/* a) 确保匹配到完整 token：前、后一位不能是标识符字符 */
		if ((p == ptx || !IS_IDCHAR(p[-1])) &&
		    !IS_IDCHAR(p[blen])) {

			/* b) 回溯跳过空白，看前一个 token 是否 .func / .entry */
			const char *q = p;
			while (q > ptx && isspace((unsigned char)q[-1]))
				--q;

			if (q - 5 >= ptx && strncmp(q - 5, ".func", 5) == 0)
				return (long)(p - ptx);   /* 命中 .func */
			if (q - 6 >= ptx && strncmp(q - 6, ".entry", 6) == 0)
				return (long)(p - ptx);   /* 命中 .entry */
		    }
		/* 继续向后搜索（+1 避免死循环） */
		++p;
	}
	return -ENOENT;      /* 未找到 */
#undef IS_IDCHAR
}

static int bpf_prog_attach_with_override(int prog_fd, const char *binary_path,
					 const char *name, int type)
{
	int offset;
	/* ---------- ① CUDA PTX 路径 ---------- */
	if (is_cuda_symbol(name)) {
		struct ptx_fd ptx;
		char ptx_path[16];
		snprintf(ptx_path, sizeof(ptx_path), "%s.ptx", binary_path);
		ptx_open(ptx_path, &ptx);
		offset = ptx_find_func_offset(ptx.ptx, name);
		ptx_close(&ptx);  /* 直接返回，无需再走 ELF */
		if (offset < 0) {
			return offset;
		}
		// set some variable
	} else {
	 	offset = elf_find_func_offset_from_file(binary_path, name);
		if (offset < 0) {
			return offset;
		}
		printf("offset: %d", offset);
		int res =
			perf_event_open_with_override(binary_path, offset, -1, 0, type);
		if (res < 0) {
			printf("perf_event_open_error_inject failed: %d\n", res);
			return res;
		}
		res = bpf_prog_attach(prog_fd, res, BPF_MODIFY_RETURN, 0);
		if (res < 0) {
			printf("bpf_prog_attach failed: %d\n", res);
			return res;
		}
	}

	return 0;
}

static int bpf_prog_attach_uprobe_with_override(int prog_fd,
						const char *binary_path,
						const char *name)
{
	return bpf_prog_attach_with_override(prog_fd, binary_path, name,
					     BPF_TYPE_UPROBE_OVERRIDE);
}

#endif // BPFTIME_UREPLACE_ATTACH_H


#endif /* D951AEE1_B02B_47EF_A451_A69982F0386C */
