/* SPDX-License-Identifier: MIT
 *
 * bpftime-macos-exec - binfmt_misc interpreter for Mach-O binaries inside
 * the bpftime ARM64 Linux VM.
 *
 * The kernel invokes this program when bpftrace/bcc (or any other tool,
 * e.g. `bpftrace -c /Users/.../app`) execve(2)s a Mach-O 64 or fat binary.
 * It asks the macOS bpftime-darwin-host to spawn the real native workload
 * (held at its entry until probes are armed), records the VM pid -> host pid
 * mapping that the syscall-server bridge uses for uprobe targeting, then
 * streams the workload's stdout/stderr and exits with its exit status.
 *
 * Wire framing mirrors darwin_bridge_protocol.md:
 *   uint32 json_len, uint32 blob_len (little endian) + JSON + blob.
 *
 * Built inside the VM with the system toolchain:
 *   gcc -O2 -Wall -Wextra bpftime_macos_exec.c -o bpftime-macos-exec
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define PIDMAP_DIR "/dev/shm/bpftime-macos-pidmap"
#define DEFAULT_PORT 49512

static int write_all(int fd, const void *buf, size_t len)
{
	const uint8_t *p = buf;
	while (len) {
		ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
		if (n < 0) {
			if (errno == EINTR)
				continue;
			return -1;
		}
		p += n;
		len -= (size_t)n;
	}
	return 0;
}

static int read_all(int fd, void *buf, size_t len)
{
	uint8_t *p = buf;
	while (len) {
		ssize_t n = recv(fd, p, len, 0);
		if (n <= 0) {
			if (n < 0 && errno == EINTR)
				continue;
			return -1;
		}
		p += n;
		len -= (size_t)n;
	}
	return 0;
}

static int send_frame(int fd, const char *json, const void *blob,
		      size_t blob_len)
{
	uint32_t hdr[2] = { (uint32_t)strlen(json), (uint32_t)blob_len };
	if (write_all(fd, hdr, sizeof(hdr)) < 0)
		return -1;
	if (hdr[0] && write_all(fd, json, hdr[0]) < 0)
		return -1;
	if (blob_len && write_all(fd, blob, blob_len) < 0)
		return -1;
	return 0;
}

static int recv_frame(int fd, char **json_out, void **blob_out,
		      size_t *blob_len)
{
	uint32_t hdr[2];
	if (read_all(fd, hdr, sizeof(hdr)) < 0)
		return -1;
	char *j = malloc(hdr[0] + 1);
	void *b = hdr[1] ? malloc(hdr[1]) : NULL;
	if (!j || (hdr[1] && !b))
		return -1;
	if (hdr[0] && read_all(fd, j, hdr[0]) < 0)
		return -1;
	j[hdr[0]] = '\0';
	if (hdr[1] && read_all(fd, b, hdr[1]) < 0)
		return -1;
	*json_out = j;
	*blob_out = b;
	*blob_len = hdr[1];
	return 0;
}

/* Minimal JSON field extraction, sufficient for the host's replies. */
static const char *json_find(const char *json, const char *key)
{
	char pattern[128];
	snprintf(pattern, sizeof(pattern), "\"%s\"", key);
	const char *p = strstr(json, pattern);
	if (!p)
		return NULL;
	p = strchr(p + strlen(pattern), ':');
	if (!p)
		return NULL;
	p++;
	while (*p == ' ' || *p == '\t')
		p++;
	return p;
}

static long json_int(const char *json, const char *key, long dflt)
{
	const char *p = json_find(json, key);
	if (!p)
		return dflt;
	char *end;
	long v = strtol(p, &end, 10);
	return end == p ? dflt : v;
}

static bool json_bool(const char *json, const char *key, bool dflt)
{
	const char *p = json_find(json, key);
	if (!p)
		return dflt;
	if (strncmp(p, "true", 4) == 0)
		return true;
	if (strncmp(p, "false", 5) == 0)
		return false;
	return dflt;
}

static int connect_host(void)
{
	const char *spec = getenv("BPFTIME_DARWIN_HOST");
	if (!spec || !*spec) {
		fprintf(stderr,
			"bpftime-macos-exec: BPFTIME_DARWIN_HOST unset\n");
		return -1;
	}
	char host[256];
	int port = DEFAULT_PORT;
	const char *colon = strrchr(spec, ':');
	if (!colon) {
		snprintf(host, sizeof(host), "%s", spec);
	} else {
		size_t n = (size_t)(colon - spec);
		if (n >= sizeof(host))
			return -1;
		memcpy(host, spec, n);
		host[n] = '\0';
		port = atoi(colon + 1);
		if (port <= 0)
			port = DEFAULT_PORT;
	}
	int fd = socket(AF_INET, SOCK_STREAM, 0);
	if (fd < 0)
		return -1;
	struct sockaddr_in addr = { 0 };
	addr.sin_family = AF_INET;
	addr.sin_port = htons((uint16_t)port);
	if (inet_pton(AF_INET, host, &addr.sin_addr) != 1 ||
	    connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
		close(fd);
		return -1;
	}
	return fd;
}

static size_t json_append_env(char *buf, size_t cap)
{
	size_t off = strlen(buf);
	off += (size_t)snprintf(buf + off, cap - off, ",\"env\":{");
	extern char **environ;
	bool first = true;
	for (char **e = environ; *e; ++e) {
		const char *eq = strchr(*e, '=');
		if (!eq)
			continue;
		off += (size_t)snprintf(buf + off, cap - off,
					"%s\"", first ? "" : ",");
		first = false;
		for (const char *q = *e; q < eq; ++q)
			off += (size_t)snprintf(buf + off, cap - off,
						"%c", *q);
		off += (size_t)snprintf(buf + off, cap - off, "\":\"");
		for (const char *q = eq + 1; *q; ++q) {
			if (*q == '"' || *q == '\\')
				off += (size_t)snprintf(buf + off, cap - off,
							"\\");
			off += (size_t)snprintf(buf + off, cap - off,
						"%c", *q);
		}
		off += (size_t)snprintf(buf + off, cap - off, "\"");
		if (off >= cap - 64)
			break;
	}
	off += (size_t)snprintf(buf + off, cap - off, "}");
	return off;
}

int main(int argc, char **argv)
{
	/* binfmt_misc passes: argv[0]=this interpreter, argv[1]=macho path,
	 * argv[2..]=original program arguments. */
	if (argc < 2) {
		fprintf(stderr,
			"usage: bpftime-macos-exec <mach-o> [args...]\n");
		return 127;
	}
	int fd = connect_host();
	if (fd < 0) {
		perror("bpftime-macos-exec: connect darwin host");
		return 126;
	}
	char *json = malloc(1u << 20);
	if (!json)
		return 126;
	size_t cap = 1u << 20;
	size_t off = (size_t)snprintf(
		json, cap,
		"{\"op\":\"spawn_cmd\",\"hold_at_entry\":true,\"cwd\":\"");
	char cwd[4096];
	if (getcwd(cwd, sizeof(cwd))) {
		for (const char *q = cwd; *q; ++q) {
			if (*q == '"' || *q == '\\')
				off += (size_t)snprintf(json + off, cap - off,
							"\\");
			off += (size_t)snprintf(json + off, cap - off,
						"%c", *q);
		}
	}
	off += (size_t)snprintf(json + off, cap - off, "\",\"argv\":[");
	for (int i = 1; i < argc; ++i) {
		off += (size_t)snprintf(json + off, cap - off,
					"%s\"", i == 1 ? "" : ",");
		for (const char *q = argv[i]; *q; ++q) {
			if (*q == '"' || *q == '\\')
				off += (size_t)snprintf(json + off, cap - off,
							"\\");
			off += (size_t)snprintf(json + off, cap - off,
						"%c", *q);
		}
		off += (size_t)snprintf(json + off, cap - off, "\"");
	}
	off += (size_t)snprintf(json + off, cap - off, "]");
	off = json_append_env(json, cap);
	off += (size_t)snprintf(json + off, cap - off,
				",\"vm_pid\":%d}", (int)getpid());

	if (send_frame(fd, json, NULL, 0) < 0) {
		perror("bpftime-macos-exec: send spawn_cmd");
		return 126;
	}
	char *resp_json = NULL;
	void *blob = NULL;
	size_t blob_len;
	if (recv_frame(fd, &resp_json, &blob, &blob_len) < 0 ||
	    !json_bool(resp_json, "ok", false)) {
		fprintf(stderr, "bpftime-macos-exec: spawn rejected: %s\n",
			resp_json ? resp_json : "(no response)");
		return 126;
	}
	long spawn_id = json_int(resp_json, "spawn_id", -1);
	long host_pid = json_int(resp_json, "host_pid", -1);
	free(resp_json);
	free(blob);
	if (spawn_id < 0 || host_pid < 0)
		return 126;

	mkdir(PIDMAP_DIR, 0777);
	char map_path[128];
	snprintf(map_path, sizeof(map_path), "%s/%d", PIDMAP_DIR,
		 (int)getpid());
	int mfd = open(map_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	if (mfd >= 0) {
		char line[64];
		int n = snprintf(line, sizeof(line), "%ld %ld\n", host_pid,
				 spawn_id);
		(void)write(mfd, line, (size_t)n);
		close(mfd);
	}

	int exit_code = 0;
	for (;;) {
		char ctl[256];
		snprintf(ctl, sizeof(ctl),
			 "{\"op\":\"cmd_ctl\",\"spawn_id\":%ld,"
			 "\"action\":\"wait_poll\",\"sig\":0}",
			 spawn_id);
		if (send_frame(fd, ctl, NULL, 0) < 0)
			break;
		char *cj = NULL;
		void *cb = NULL;
		size_t cbl;
		if (recv_frame(fd, &cj, &cb, &cbl) < 0)
			break;
		if (cbl) {
			size_t done = 0;
			while (done < cbl) {
				ssize_t n = write(STDOUT_FILENO,
						  (char *)cb + done, cbl - done);
				if (n <= 0)
					break;
				done += (size_t)n;
			}
		}
		bool exited = json_bool(cj, "exited", false);
		exit_code = (int)json_int(cj, "exit_code", 0);
		long term_sig = json_int(cj, "term_sig", 0);
		free(cj);
		free(cb);
		if (exited) {
			if (term_sig)
				exit_code = 128 + (int)term_sig;
			break;
		}
		usleep(20 * 1000);
	}
	unlink(map_path);
	close(fd);
	return exit_code & 0xff;
}
