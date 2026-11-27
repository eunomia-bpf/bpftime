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
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>

#ifndef PIDMAP_DIR
#define PIDMAP_DIR "/dev/shm/bpftime-macos-pidmap"
#endif
#define DEFAULT_PORT 49512
#ifndef JSON_CAP
#define JSON_CAP (1u << 20) /* max spawn-request JSON: 1 MiB */
#endif
#define RECV_JSON_MAX (1u << 20)  /* max host JSON frame: 1 MiB */
#define RECV_BLOB_MAX (16u << 20) /* max host stdout blob: 16 MiB */
#define IO_TIMEOUT_SEC 30         /* bounded socket waits */

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
			return -1; /* EOF, error, or SO_RCVTIMEO timeout */
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

/* Validate advertised lengths before allocating or receiving, so a hostile
 * 0xFFFFFFFF length cannot wrap the allocation or drive a huge read. */
static int recv_frame(int fd, char **json_out, void **blob_out,
		      size_t *blob_len)
{
	uint32_t hdr[2];
	if (read_all(fd, hdr, sizeof(hdr)) < 0)
		return -1;
	if (hdr[0] > RECV_JSON_MAX || hdr[1] > RECV_BLOB_MAX)
		return -1;
	char *j = malloc((size_t)hdr[0] + 1);
	void *b = hdr[1] ? malloc((size_t)hdr[1]) : NULL;
	if (!j || (hdr[1] && !b)) {
		free(j);
		free(b);
		return -1;
	}
	if (hdr[0] && read_all(fd, j, hdr[0]) < 0)
		goto fail;
	j[hdr[0]] = '\0';
	if (hdr[1] && read_all(fd, b, hdr[1]) < 0)
		goto fail;
	*json_out = j;
	*blob_out = b;
	*blob_len = hdr[1];
	return 0;
fail:
	free(j);
	free(b);
	return -1;
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

/* Bounded JSON buffer: every append is capacity-checked and sets err instead
 * of advancing an offset past the allocation (which would underflow the
 * remaining-size arithmetic and overflow the heap). */
struct jsonbuf {
	char *p;
	size_t cap;
	size_t off;
	int err;
};

static void jb_put(struct jsonbuf *j, const char *s)
{
	size_t n;
	if (j->err)
		return;
	n = strlen(s);
	if (n >= j->cap - j->off) {
		j->err = 1;
		return;
	}
	memcpy(j->p + j->off, s, n);
	j->off += n;
	j->p[j->off] = '\0';
}

static void jb_printf(struct jsonbuf *j, const char *fmt, ...)
{
	va_list ap;
	int r;
	if (j->err)
		return;
	va_start(ap, fmt);
	r = vsnprintf(j->p + j->off, j->cap - j->off, fmt, ap);
	va_end(ap);
	if (r < 0 || (size_t)r >= j->cap - j->off) {
		j->err = 1;
		return;
	}
	j->off += (size_t)r;
}

/* Append one fully-escaped JSON string body (without enclosing quotes):
 * quotes, backslashes and every control character are encoded, so multiline
 * environment values or control bytes in argv/cwd cannot break the frame. */
static void jb_escape(struct jsonbuf *j, const char *s)
{
	static const char hexd[] = "0123456789abcdef";
	if (j->err)
		return;
	for (const uint8_t *q = (const uint8_t *)s; *q; ++q) {
		char out[8];
		const char *e;
		size_t n;
		switch (*q) {
		case '"':
			e = "\\\"";
			break;
		case '\\':
			e = "\\\\";
			break;
		case '\b':
			e = "\\b";
			break;
		case '\f':
			e = "\\f";
			break;
		case '\n':
			e = "\\n";
			break;
		case '\r':
			e = "\\r";
			break;
		case '\t':
			e = "\\t";
			break;
		default:
			if (*q < 0x20) {
				out[0] = '\\';
				out[1] = 'u';
				out[2] = '0';
				out[3] = '0';
				out[4] = hexd[(*q >> 4) & 0xf];
				out[5] = hexd[*q & 0xf];
				out[6] = '\0';
				e = out;
			} else {
				out[0] = (char)*q;
				out[1] = '\0';
				e = out;
			}
			break;
		}
		n = strlen(e);
		if (n >= j->cap - j->off) {
			j->err = 1;
			return;
		}
		memcpy(j->p + j->off, e, n);
		j->off += n;
	}
	j->p[j->off] = '\0';
}

/* Escape a fixed-length key without mutating the environment entry. */
static void jb_escape_n(struct jsonbuf *j, const char *s, size_t len)
{
	char key[256];
	if (len >= sizeof(key)) {
		j->err = 1;
		return;
	}
	memcpy(key, s, len);
	key[len] = '\0';
	jb_escape(j, key);
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
	/* Bound every wait so a bridge that accepts then stalls cannot hang the
	 * invoking tool indefinitely; a timeout surfaces as a transport error. */
	struct timeval tv = { .tv_sec = IO_TIMEOUT_SEC, .tv_usec = 0 };
	setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
	setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
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

static volatile sig_atomic_t cancel_requested;

static void on_cancel(int sig)
{
	(void)sig;
	cancel_requested = 1;
}

static void install_cancel_handlers(void)
{
	struct sigaction sa = { 0 };
	sa.sa_handler = on_cancel;
	sigemptyset(&sa.sa_mask);
	sigaction(SIGINT, &sa, NULL);
	sigaction(SIGTERM, &sa, NULL);
	sigaction(SIGHUP, &sa, NULL);
	signal(SIGPIPE, SIG_IGN);
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

	char *raw = malloc(JSON_CAP);
	if (!raw)
		return 126;
	struct jsonbuf jb = { .p = raw, .cap = JSON_CAP, .off = 0, .err = 0 };
	jb_put(&jb, "{\"op\":\"spawn_cmd\",\"hold_at_entry\":true,\"cwd\":\"");
	char cwd[4096];
	if (getcwd(cwd, sizeof(cwd)))
		jb_escape(&jb, cwd);
	jb_put(&jb, "\",\"argv\":[");
	for (int i = 1; i < argc; ++i) {
		if (i > 1)
			jb_put(&jb, ",");
		jb_put(&jb, "\"");
		jb_escape(&jb, argv[i]);
		jb_put(&jb, "\"");
	}
	jb_put(&jb, "],\"env\":{");
	bool first = true;
	extern char **environ;
	for (char **e = environ; *e; ++e) {
		const char *eq = strchr(*e, '=');
		if (!eq)
			continue;
		if (!first)
			jb_put(&jb, ",");
		first = false;
		jb_put(&jb, "\"");
		/* key spans [*e, eq) and is not NUL-terminated in place */
		jb_escape_n(&jb, *e, (size_t)(eq - *e));
		jb_put(&jb, "\":\"");
		jb_escape(&jb, eq + 1);
		jb_put(&jb, "\"");
	}
	jb_put(&jb, "},\"vm_pid\":");
	jb_printf(&jb, "%d", (int)getpid());
	jb_put(&jb, "}");
	if (jb.err) {
		fprintf(stderr,
			"bpftime-macos-exec: spawn request exceeds %u bytes\n",
			JSON_CAP);
		free(raw);
		close(fd);
		return 126;
	}

	if (send_frame(fd, raw, NULL, 0) < 0) {
		perror("bpftime-macos-exec: send spawn_cmd");
		free(raw);
		close(fd);
		return 126;
	}
	free(raw);

	char *resp_json = NULL;
	void *blob = NULL;
	size_t blob_len;
	if (recv_frame(fd, &resp_json, &blob, &blob_len) < 0 ||
	    !json_bool(resp_json, "ok", false)) {
		fprintf(stderr, "bpftime-macos-exec: spawn rejected: %s\n",
			resp_json ? resp_json : "(no response)");
		free(resp_json);
		free(blob);
		close(fd);
		return 126;
	}
	long spawn_id = json_int(resp_json, "spawn_id", -1);
	long host_pid = json_int(resp_json, "host_pid", -1);
	free(resp_json);
	free(blob);
	if (spawn_id < 0 || host_pid < 0)
		return 126;

	install_cancel_handlers();

	/* The shared pidmap must stay writable for later non-sudo callers:
	 * make it world-writable + sticky (like /tmp). Failure to record the
	 * mapping means the bridge cannot target this workload, so cancel it. */
	mkdir(PIDMAP_DIR, 0777);
	chmod(PIDMAP_DIR, 01777);
	char map_path[128];
	snprintf(map_path, sizeof(map_path), "%s/%d", PIDMAP_DIR,
		 (int)getpid());
	int mfd = open(map_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	if (mfd < 0 ||
	    dprintf(mfd, "%ld %ld\n", host_pid, spawn_id) < 0) {
		char kill[160];
		snprintf(kill, sizeof(kill),
			 "{\"op\":\"cmd_ctl\",\"spawn_id\":%ld,"
			 "\"action\":\"kill\",\"sig\":%d}",
			 spawn_id, SIGTERM);
		(void)send_frame(fd, kill, NULL, 0);
		close(mfd);
		close(fd);
		return 126;
	}
	close(mfd);

	int exit_code = 126; /* failure unless an exited frame is received */
	bool got_exit = false;
	bool kill_sent = false;
	for (;;) {
		if (cancel_requested && !kill_sent) {
			/* Forward Ctrl-C / termination to the native workload so
			 * it does not outlive the interpreter or stay held. */
			char kill[160];
			snprintf(kill, sizeof(kill),
				 "{\"op\":\"cmd_ctl\",\"spawn_id\":%ld,"
				 "\"action\":\"kill\",\"sig\":%d}",
				 spawn_id, SIGTERM);
			if (send_frame(fd, kill, NULL, 0) < 0)
				break;
			kill_sent = true;
			cancel_requested = 0;
		}
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
		long term_sig = json_int(cj, "term_sig", 0);
		if (exited) {
			exit_code = (int)json_int(cj, "exit_code", 0);
			if (term_sig)
				exit_code = 128 + (int)term_sig;
			got_exit = true;
		}
		free(cj);
		free(cb);
		if (exited)
			break;
		usleep(20 * 1000);
	}
	unlink(map_path);
	close(fd);
	/* A lost bridge or malformed reply must never look like success. */
	return got_exit ? (exit_code & 0xff) : 126;
}
