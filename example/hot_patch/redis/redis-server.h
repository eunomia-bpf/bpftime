#ifndef __VMLINUX_REDIS_H__
#define __VMLINUX_REDIS_H__

#ifndef BPF_NO_PRESERVE_ACCESS_INDEX
#pragma clang attribute push (__attribute__((preserve_access_index)), apply_to = record)
#endif

// typedef long unsigned int size_t;

struct listNode {
	struct listNode *prev;
	struct listNode *next;
	void *value;
};

typedef struct listNode listNode;

struct listIter {
	listNode *next;
	int direction;
};

typedef struct listIter listIter;

struct list {
	listNode *head;
	listNode *tail;
	void * (*dup)(void *);
	void (*free)(void *);
	int (*match)(void *, void *);
	long unsigned int len;
};

typedef struct list list;

struct quicklistNode {
	struct quicklistNode *prev;
	struct quicklistNode *next;
	unsigned char *zl;
	unsigned int sz;
	unsigned int count: 16;
	unsigned int encoding: 2;
	unsigned int container: 2;
	unsigned int recompress: 1;
	unsigned int attempted_compress: 1;
	unsigned int extra: 10;
};

typedef struct quicklistNode quicklistNode;

struct quicklistLZF {
	unsigned int sz;
	char compressed[0];
};

typedef struct quicklistLZF quicklistLZF;

struct quicklist {
	quicklistNode *head;
	quicklistNode *tail;
	long unsigned int count;
	long unsigned int len;
	int fill: 16;
	unsigned int compress: 16;
};

typedef struct quicklist quicklist;

struct quicklistIter {
	const quicklist *quicklist;
	quicklistNode *current;
	unsigned char *zi;
	long int offset;
	int direction;
};

typedef struct quicklistIter quicklistIter;

struct quicklistEntry {
	const quicklist *quicklist;
	quicklistNode *node;
	unsigned char *zi;
	unsigned char *value;
	long long int longval;
	unsigned int sz;
	int offset;
};

typedef struct quicklistEntry quicklistEntry;

typedef unsigned int __uint32_t;

typedef long unsigned int __uint64_t;

typedef long int __time_t;

typedef long int __suseconds_t;

typedef __time_t time_t;

struct timeval {
	__time_t tv_sec;
	__suseconds_t tv_usec;
};

typedef long unsigned int nfds_t;

struct aeEventLoop;

typedef void aeFileProc(struct aeEventLoop *, int, void *, int);

typedef void aeBeforeSleepProc(struct aeEventLoop *);

struct aeFileEvent;

typedef struct aeFileEvent aeFileEvent;

struct aeFiredEvent;

typedef struct aeFiredEvent aeFiredEvent;

struct aeTimeEvent;

typedef struct aeTimeEvent aeTimeEvent;

struct aeEventLoop {
	int maxfd;
	int setsize;
	long long int timeEventNextId;
	time_t lastTime;
	aeFileEvent *events;
	aeFiredEvent *fired;
	aeTimeEvent *timeEventHead;
	int stop;
	void *apidata;
	aeBeforeSleepProc *beforesleep;
	aeBeforeSleepProc *aftersleep;
};

typedef int aeTimeProc(struct aeEventLoop *, long long int, void *);

typedef void aeEventFinalizerProc(struct aeEventLoop *, void *);

struct aeFileEvent {
	int mask;
	aeFileProc *rfileProc;
	aeFileProc *wfileProc;
	void *clientData;
};

struct aeTimeEvent {
	long long int id;
	long int when_sec;
	long int when_ms;
	aeTimeProc *timeProc;
	aeEventFinalizerProc *finalizerProc;
	void *clientData;
	struct aeTimeEvent *prev;
	struct aeTimeEvent *next;
};

struct aeFiredEvent {
	int fd;
	int mask;
};

typedef struct aeEventLoop aeEventLoop;

enum EPOLL_EVENTS {
	EPOLLIN = 1,
	EPOLLPRI = 2,
	EPOLLOUT = 4,
	EPOLLRDNORM = 64,
	EPOLLRDBAND = 128,
	EPOLLWRNORM = 256,
	EPOLLWRBAND = 512,
	EPOLLMSG = 1024,
	EPOLLERR = 8,
	EPOLLHUP = 16,
	EPOLLRDHUP = 8192,
	EPOLLEXCLUSIVE = 268435456,
	EPOLLWAKEUP = 536870912,
	EPOLLONESHOT = 1073741824,
	EPOLLET = 2147483648,
};

union epoll_data {
	void *ptr;
	int fd;
	uint32_t u32;
	uint64_t u64;
};

typedef union epoll_data epoll_data_t;

struct aeApiState {
	int epfd;
	struct epoll_event *events;
};

typedef struct aeApiState aeApiState;

typedef unsigned char __uint8_t;

typedef short unsigned int __uint16_t;

typedef unsigned int __mode_t;

typedef long int __ssize_t;

typedef unsigned int __socklen_t;

typedef __mode_t mode_t;

// typedef __ssize_t ssize_t;

typedef __socklen_t socklen_t;

typedef short unsigned int sa_family_t;


struct sockaddr_storage {
	sa_family_t ss_family;
	char __ss_padding[118];
	long unsigned int __ss_align;
};

struct sockaddr_at;

struct sockaddr_ax25;

struct sockaddr_dl;

struct sockaddr_eon;

typedef __uint16_t uint16_t;

typedef uint16_t in_port_t;

typedef uint32_t in_addr_t;

typedef __uint8_t uint8_t;

struct sockaddr_inarp;

struct sockaddr_ipx;

struct sockaddr_iso;

struct sockaddr_ns;

struct sockaddr_x25;

typedef union {
} __SOCKADDR_ARG;

typedef union {
} __CONST_SOCKADDR_ARG;

struct addrinfo {
	int ai_flags;
	int ai_family;
	int ai_socktype;
	int ai_protocol;
	socklen_t ai_addrlen;
	struct sockaddr *ai_addr;
	char *ai_canonname;
	struct addrinfo *ai_next;
};

typedef __builtin_va_list __gnuc_va_list;

// typedef __gnuc_va_list va_list;


struct dictEntry {
	void *key;
	union {
		void *val;
		uint64_t u64;
		int64_t s64;
		double d;
	} v;
	struct dictEntry *next;
};

typedef struct dictEntry dictEntry;

struct dictType {
	uint64_t (*hashFunction)(const void *);
	void * (*keyDup)(void *, const void *);
	void * (*valDup)(void *, const void *);
	int (*keyCompare)(void *, const void *, const void *);
	void (*keyDestructor)(void *, void *);
	void (*valDestructor)(void *, void *);
};

typedef struct dictType dictType;

struct dictht {
	dictEntry **table;
	long unsigned int size;
	long unsigned int sizemask;
	long unsigned int used;
};

typedef struct dictht dictht;

struct dict {
	dictType *type;
	void *privdata;
	dictht ht[2];
	long int rehashidx;
	long unsigned int iterators;
};

typedef struct dict dict;

struct dictIterator {
	dict *d;
	long int index;
	int table;
	int safe;
	dictEntry *entry;
	dictEntry *nextEntry;
	long long int fingerprint;
};

typedef struct dictIterator dictIterator;

typedef void dictScanFunction(void *, const dictEntry *);

typedef void dictScanBucketFunction(void *, dictEntry **);

typedef int __int32_t;

typedef long int __intmax_t;

typedef unsigned int __uid_t;

typedef long int __off_t;

typedef long int __off64_t;

typedef int __pid_t;

typedef long int __clock_t;

typedef long unsigned int __rlim64_t;

typedef unsigned int __useconds_t;

typedef long int __syscall_slong_t;

typedef __off64_t off_t;

typedef __pid_t pid_t;

struct __pthread_internal_list {
	struct __pthread_internal_list *__prev;
	struct __pthread_internal_list *__next;
};

typedef struct __pthread_internal_list __pthread_list_t;

struct __pthread_mutex_s {
	int __lock;
	unsigned int __count;
	int __owner;
	unsigned int __nusers;
	int __kind;
	short int __spins;
	short int __elision;
	__pthread_list_t __list;
};

typedef union {
	char __size[4];
	int __align;
} pthread_mutexattr_t;

typedef union {
	struct __pthread_mutex_s __data;
	char __size[40];
	long int __align;
} pthread_mutex_t;

struct _IO_marker;

typedef void _IO_lock_t;

struct _IO_codecvt;

struct _IO_wide_data;

struct _IO_FILE {
	int _flags;
	char *_IO_read_ptr;
	char *_IO_read_end;
	char *_IO_read_base;
	char *_IO_write_base;
	char *_IO_write_ptr;
	char *_IO_write_end;
	char *_IO_buf_base;
	char *_IO_buf_end;
	char *_IO_save_base;
	char *_IO_backup_base;
	char *_IO_save_end;
	struct _IO_marker *_markers;
	struct _IO_FILE *_chain;
	int _fileno;
	int _flags2;
	__off_t _old_offset;
	short unsigned int _cur_column;
	signed char _vtable_offset;
	char _shortbuf[1];
	_IO_lock_t *_lock;
	__off64_t _offset;
	struct _IO_codecvt *_codecvt;
	struct _IO_wide_data *_wide_data;
	struct _IO_FILE *_freeres_list;
	void *_freeres_buf;
	size_t __pad5;
	int _mode;
	char _unused2[20];
};

typedef struct _IO_FILE FILE;

typedef __intmax_t intmax_t;

typedef char *sds;

struct sdshdr8 {
	uint8_t len;
	uint8_t alloc;
	unsigned char flags;
	char buf[0];
};

struct sdshdr16 {
	uint16_t len;
	uint16_t alloc;
	unsigned char flags;
	char buf[0];
} __attribute__((packed));

struct sdshdr32 {
	uint32_t len;
	uint32_t alloc;
	unsigned char flags;
	char buf[0];
} __attribute__((packed));

struct sdshdr64 {
	uint64_t len;
	uint64_t alloc;
	unsigned char flags;
	char buf[0];
} __attribute__((packed));

enum {
	MSG_OOB = 1,
	MSG_PEEK = 2,
	MSG_DONTROUTE = 4,
	MSG_TRYHARD = 4,
	MSG_CTRUNC = 8,
	MSG_PROXY = 16,
	MSG_TRUNC = 32,
	MSG_DONTWAIT = 64,
	MSG_EOR = 128,
	MSG_WAITALL = 256,
	MSG_FIN = 512,
	MSG_SYN = 1024,
	MSG_CONFIRM = 2048,
	MSG_RST = 4096,
	MSG_ERRQUEUE = 8192,
	MSG_NOSIGNAL = 16384,
	MSG_MORE = 32768,
	MSG_WAITFORONE = 65536,
	MSG_BATCH = 262144,
	MSG_ZEROCOPY = 67108864,
	MSG_FASTOPEN = 536870912,
	MSG_CMSG_CLOEXEC = 1073741824,
};

struct lua_State;

typedef struct lua_State lua_State;

typedef long long int mstime_t;

struct raxNode {
	uint32_t iskey: 1;
	uint32_t isnull: 1;
	uint32_t iscompr: 1;
	uint32_t size: 29;
	unsigned char data[0];
};

typedef struct raxNode raxNode;

struct rax {
	raxNode *head;
	uint64_t numele;
	uint64_t numnodes;
};

typedef struct rax rax;

struct redisObject {
	unsigned int type: 4;
	unsigned int encoding: 4;
	unsigned int lru: 24;
	int refcount;
	void *ptr;
};

typedef struct redisObject robj;

struct redisDb {
	dict *dict;
	dict *expires;
	dict *blocking_keys;
	dict *ready_keys;
	dict *watched_keys;
	int id;
	long long int avg_ttl;
	list *defrag_later;
};

typedef struct redisDb redisDb;

struct redisCommand;

struct multiCmd {
	robj **argv;
	int argc;
	struct redisCommand *cmd;
};

struct client;

typedef struct client client;

typedef void redisCommandProc(client *);

typedef int *redisGetKeysProc(struct redisCommand *, robj **, int, int *);

struct redisCommand {
	char *name;
	redisCommandProc *proc;
	int arity;
	char *sflags;
	int flags;
	redisGetKeysProc *getkeys_proc;
	int firstkey;
	int lastkey;
	int keystep;
	long long int microseconds;
	long long int calls;
};

typedef struct multiCmd multiCmd;

struct multiState {
	multiCmd *commands;
	int count;
	int minreplicas;
	time_t minreplicas_timeout;
};

typedef struct multiState multiState;

struct blockingState {
	mstime_t timeout;
	dict *keys;
	robj *target;
	size_t xread_count;
	robj *xread_group;
	robj *xread_consumer;
	mstime_t xread_retry_time;
	mstime_t xread_retry_ttl;
	int numreplicas;
	long long int reploffset;
	void *module_blocked_handle;
};

typedef struct blockingState blockingState;

struct client {
	uint64_t id;
	int fd;
	redisDb *db;
	robj *name;
	sds querybuf;
	sds pending_querybuf;
	size_t querybuf_peak;
	int argc;
	robj **argv;
	struct redisCommand *cmd;
	struct redisCommand *lastcmd;
	int reqtype;
	int multibulklen;
	long int bulklen;
	list *reply;
	long long unsigned int reply_bytes;
	size_t sentlen;
	time_t ctime;
	time_t lastinteraction;
	time_t obuf_soft_limit_reached_time;
	int flags;
	int authenticated;
	int replstate;
	int repl_put_online_on_ack;
	int repldbfd;
	off_t repldboff;
	off_t repldbsize;
	sds replpreamble;
	long long int read_reploff;
	long long int reploff;
	long long int repl_ack_off;
	long long int repl_ack_time;
	long long int psync_initial_offset;
	char replid[41];
	int slave_listening_port;
	char slave_ip[46];
	int slave_capa;
	multiState mstate;
	int btype;
	blockingState bpop;
	long long int woff;
	list *watched_keys;
	dict *pubsub_channels;
	list *pubsub_patterns;
	sds peerid;
	listNode *client_list_node;
	int bufpos;
	char buf[16384];
};

struct saveparam {
	time_t seconds;
	int changes;
};

struct sharedObjectsStruct {
	robj *crlf;
	robj *ok;
	robj *err;
	robj *emptybulk;
	robj *czero;
	robj *cone;
	robj *cnegone;
	robj *pong;
	robj *space;
	robj *colon;
	robj *nullbulk;
	robj *nullmultibulk;
	robj *queued;
	robj *emptymultibulk;
	robj *wrongtypeerr;
	robj *nokeyerr;
	robj *syntaxerr;
	robj *sameobjecterr;
	robj *outofrangeerr;
	robj *noscripterr;
	robj *loadingerr;
	robj *slowscripterr;
	robj *bgsaveerr;
	robj *masterdownerr;
	robj *roslaveerr;
	robj *execaborterr;
	robj *noautherr;
	robj *noreplicaserr;
	robj *busykeyerr;
	robj *oomerr;
	robj *plus;
	robj *messagebulk;
	robj *pmessagebulk;
	robj *subscribebulk;
	robj *unsubscribebulk;
	robj *psubscribebulk;
	robj *punsubscribebulk;
	robj *del;
	robj *unlink;
	robj *rpop;
	robj *lpop;
	robj *lpush;
	robj *zpopmin;
	robj *zpopmax;
	robj *emptyscan;
	robj *select[10];
	robj *integers[10000];
	robj *mbulkhdr[32];
	robj *bulkhdr[32];
	sds minstring;
	sds maxstring;
};

struct clientBufferLimitsConfig {
	long long unsigned int hard_limit_bytes;
	long long unsigned int soft_limit_bytes;
	time_t soft_limit_seconds;
};

typedef struct clientBufferLimitsConfig clientBufferLimitsConfig;

struct redisOp {
	robj **argv;
	int argc;
	int dbid;
	int target;
	struct redisCommand *cmd;
};

typedef struct redisOp redisOp;

struct redisOpArray {
	redisOp *ops;
	int numops;
};

typedef struct redisOpArray redisOpArray;

struct redisMemOverhead {
	size_t peak_allocated;
	size_t total_allocated;
	size_t startup_allocated;
	size_t repl_backlog;
	size_t clients_slaves;
	size_t clients_normal;
	size_t aof_buffer;
	size_t overhead_total;
	size_t dataset;
	size_t total_keys;
	size_t bytes_per_key;
	float dataset_perc;
	float peak_perc;
	float total_frag;
	size_t total_frag_bytes;
	float allocator_frag;
	size_t allocator_frag_bytes;
	float allocator_rss;
	size_t allocator_rss_bytes;
	float rss_extra;
	size_t rss_extra_bytes;
	size_t num_dbs;
	struct {
		size_t dbid;
		size_t overhead_ht_main;
		size_t overhead_ht_expires;
	} *db;
};

struct rdbSaveInfo {
	int repl_stream_db;
	int repl_id_is_set;
	char repl_id[41];
	long long int repl_offset;
};

typedef struct rdbSaveInfo rdbSaveInfo;

struct malloc_stats {
	size_t zmalloc_used;
	size_t process_rss;
	size_t allocator_allocated;
	size_t allocator_active;
	size_t allocator_resident;
};

typedef struct malloc_stats malloc_stats;

struct clusterState;

struct redisServer {
	pid_t pid;
	char *configfile;
	char *executable;
	char **exec_argv;
	int hz;
	redisDb *db;
	dict *commands;
	dict *orig_commands;
	aeEventLoop *el;
	unsigned int lruclock;
	int shutdown_asap;
	int activerehashing;
	int active_defrag_running;
	char *requirepass;
	char *pidfile;
	int arch_bits;
	int cronloops;
	char runid[41];
	int sentinel_mode;
	size_t initial_memory_usage;
	int always_show_logo;
	dict *moduleapi;
	list *loadmodule_queue;
	int module_blocked_pipe[2];
	int port;
	int tcp_backlog;
	char *bindaddr[16];
	int bindaddr_count;
	char *unixsocket;
	mode_t unixsocketperm;
	int ipfd[16];
	int ipfd_count;
	int sofd;
	int cfd[16];
	int cfd_count;
	list *clients;
	list *clients_to_close;
	list *clients_pending_write;
	list *slaves;
	list *monitors;
	client *current_client;
	int clients_paused;
	mstime_t clients_pause_end_time;
	char neterr[256];
	dict *migrate_cached_sockets;
	uint64_t next_client_id;
	int protected_mode;
	int loading;
	off_t loading_total_bytes;
	off_t loading_loaded_bytes;
	time_t loading_start_time;
	off_t loading_process_events_interval_bytes;
	struct redisCommand *delCommand;
	struct redisCommand *multiCommand;
	struct redisCommand *lpushCommand;
	struct redisCommand *lpopCommand;
	struct redisCommand *rpopCommand;
	struct redisCommand *zpopminCommand;
	struct redisCommand *zpopmaxCommand;
	struct redisCommand *sremCommand;
	struct redisCommand *execCommand;
	struct redisCommand *expireCommand;
	struct redisCommand *pexpireCommand;
	struct redisCommand *xclaimCommand;
	time_t stat_starttime;
	long long int stat_numcommands;
	long long int stat_numconnections;
	long long int stat_expiredkeys;
	double stat_expired_stale_perc;
	long long int stat_expired_time_cap_reached_count;
	long long int stat_evictedkeys;
	long long int stat_keyspace_hits;
	long long int stat_keyspace_misses;
	long long int stat_active_defrag_hits;
	long long int stat_active_defrag_misses;
	long long int stat_active_defrag_key_hits;
	long long int stat_active_defrag_key_misses;
	long long int stat_active_defrag_scanned;
	size_t stat_peak_memory;
	long long int stat_fork_time;
	double stat_fork_rate;
	long long int stat_rejected_conn;
	long long int stat_sync_full;
	long long int stat_sync_partial_ok;
	long long int stat_sync_partial_err;
	list *slowlog;
	long long int slowlog_entry_id;
	long long int slowlog_log_slower_than;
	long unsigned int slowlog_max_len;
	malloc_stats cron_malloc_stats;
	long long int stat_net_input_bytes;
	long long int stat_net_output_bytes;
	size_t stat_rdb_cow_bytes;
	size_t stat_aof_cow_bytes;
	struct {
		long long int last_sample_time;
		long long int last_sample_count;
		long long int samples[16];
		int idx;
	} inst_metric[3];
	int verbosity;
	int maxidletime;
	int tcpkeepalive;
	int active_expire_enabled;
	int active_defrag_enabled;
	size_t active_defrag_ignore_bytes;
	int active_defrag_threshold_lower;
	int active_defrag_threshold_upper;
	int active_defrag_cycle_min;
	int active_defrag_cycle_max;
	long unsigned int active_defrag_max_scan_fields;
	size_t client_max_querybuf_len;
	int dbnum;
	int supervised;
	int supervised_mode;
	int daemonize;
	clientBufferLimitsConfig client_obuf_limits[3];
	int aof_state;
	int fdatasync;
	char *aof_filename;
	int aof_no_fsync_on_rewrite;
	int aof_rewrite_perc;
	off_t aof_rewrite_min_size;
	off_t aof_rewrite_base_size;
	off_t aof_current_size;
	int aof_rewrite_scheduled;
	pid_t aof_child_pid;
	list *aof_rewrite_buf_blocks;
	sds aof_buf;
	int aof_fd;
	int aof_selected_db;
	time_t aof_flush_postponed_start;
	time_t aof_last_fsync;
	time_t aof_rewrite_time_last;
	time_t aof_rewrite_time_start;
	int aof_lastbgrewrite_status;
	long unsigned int aof_delayed_fsync;
	int aof_rewrite_incremental_fsync;
	int aof_last_write_status;
	int aof_last_write_errno;
	int aof_load_truncated;
	int aof_use_rdb_preamble;
	int aof_pipe_write_data_to_child;
	int aof_pipe_read_data_from_parent;
	int aof_pipe_write_ack_to_parent;
	int aof_pipe_read_ack_from_child;
	int aof_pipe_write_ack_to_child;
	int aof_pipe_read_ack_from_parent;
	int aof_stop_sending_diff;
	sds aof_child_diff;
	long long int dirty;
	long long int dirty_before_bgsave;
	pid_t rdb_child_pid;
	struct saveparam *saveparams;
	int saveparamslen;
	char *rdb_filename;
	int rdb_compression;
	int rdb_checksum;
	time_t lastsave;
	time_t lastbgsave_try;
	time_t rdb_save_time_last;
	time_t rdb_save_time_start;
	int rdb_bgsave_scheduled;
	int rdb_child_type;
	int lastbgsave_status;
	int stop_writes_on_bgsave_err;
	int rdb_pipe_write_result_to_parent;
	int rdb_pipe_read_result_from_child;
	int child_info_pipe[2];
	struct {
		int process_type;
		size_t cow_size;
		long long unsigned int magic;
	} child_info_data;
	redisOpArray also_propagate;
	char *logfile;
	int syslog_enabled;
	char *syslog_ident;
	int syslog_facility;
	char replid[41];
	char replid2[41];
	long long int master_repl_offset;
	long long int second_replid_offset;
	int slaveseldb;
	int repl_ping_slave_period;
	char *repl_backlog;
	long long int repl_backlog_size;
	long long int repl_backlog_histlen;
	long long int repl_backlog_idx;
	long long int repl_backlog_off;
	time_t repl_backlog_time_limit;
	time_t repl_no_slaves_since;
	int repl_min_slaves_to_write;
	int repl_min_slaves_max_lag;
	int repl_good_slaves_count;
	int repl_diskless_sync;
	int repl_diskless_sync_delay;
	char *masterauth;
	char *masterhost;
	int masterport;
	int repl_timeout;
	client *master;
	client *cached_master;
	int repl_syncio_timeout;
	int repl_state;
	off_t repl_transfer_size;
	off_t repl_transfer_read;
	off_t repl_transfer_last_fsync_off;
	int repl_transfer_s;
	int repl_transfer_fd;
	char *repl_transfer_tmpfile;
	time_t repl_transfer_lastio;
	int repl_serve_stale_data;
	int repl_slave_ro;
	time_t repl_down_since;
	int repl_disable_tcp_nodelay;
	int slave_priority;
	int slave_announce_port;
	char *slave_announce_ip;
	char master_replid[41];
	long long int master_initial_offset;
	int repl_slave_lazy_flush;
	dict *repl_scriptcache_dict;
	list *repl_scriptcache_fifo;
	unsigned int repl_scriptcache_size;
	list *clients_waiting_acks;
	int get_ack_from_slaves;
	unsigned int maxclients;
	long long unsigned int maxmemory;
	int maxmemory_policy;
	int maxmemory_samples;
	int lfu_log_factor;
	int lfu_decay_time;
	long long int proto_max_bulk_len;
	unsigned int blocked_clients;
	unsigned int blocked_clients_by_type[6];
	list *unblocked_clients;
	list *ready_keys;
	int sort_desc;
	int sort_alpha;
	int sort_bypattern;
	int sort_store;
	size_t hash_max_ziplist_entries;
	size_t hash_max_ziplist_value;
	size_t set_max_intset_entries;
	size_t zset_max_ziplist_entries;
	size_t zset_max_ziplist_value;
	size_t hll_sparse_max_bytes;
	int list_max_ziplist_size;
	int list_compress_depth;
	time_t unixtime;
	long long int mstime;
	dict *pubsub_channels;
	list *pubsub_patterns;
	int notify_keyspace_events;
	int cluster_enabled;
	mstime_t cluster_node_timeout;
	char *cluster_configfile;
	struct clusterState *cluster;
	int cluster_migration_barrier;
	int cluster_slave_validity_factor;
	int cluster_require_full_coverage;
	int cluster_slave_no_failover;
	char *cluster_announce_ip;
	int cluster_announce_port;
	int cluster_announce_bus_port;
	lua_State *lua;
	client *lua_client;
	client *lua_caller;
	dict *lua_scripts;
	mstime_t lua_time_limit;
	mstime_t lua_time_start;
	int lua_write_dirty;
	int lua_random_dirty;
	int lua_replicate_commands;
	int lua_multi_emitted;
	int lua_repl;
	int lua_timedout;
	int lua_kill;
	int lua_always_replicate_commands;
	int lazyfree_lazy_eviction;
	int lazyfree_lazy_expire;
	int lazyfree_lazy_server_del;
	long long int latency_monitor_threshold;
	dict *latency_events;
	const char *assert_failed;
	const char *assert_file;
	int assert_line;
	int bug_report_start;
	int watchdog_period;
	size_t system_memory_size;
	pthread_mutex_t lruclock_mutex;
	pthread_mutex_t next_client_id_mutex;
	pthread_mutex_t unixtime_mutex;
};

struct clusterNode;

typedef struct clusterNode clusterNode;

struct clusterState {
	clusterNode *myself;
	uint64_t currentEpoch;
	int state;
	int size;
	dict *nodes;
	dict *nodes_black_list;
	clusterNode *migrating_slots_to[16384];
	clusterNode *importing_slots_from[16384];
	clusterNode *slots[16384];
	uint64_t slots_keys_count[16384];
	rax *slots_to_keys;
	mstime_t failover_auth_time;
	int failover_auth_count;
	int failover_auth_sent;
	int failover_auth_rank;
	uint64_t failover_auth_epoch;
	int cant_failover_reason;
	mstime_t mf_end;
	clusterNode *mf_slave;
	long long int mf_master_offset;
	int mf_can_start;
	uint64_t lastVoteEpoch;
	int todo_before_sleep;
	long long int stats_bus_messages_sent[10];
	long long int stats_bus_messages_received[10];
	long long int stats_pfail_nodes;
};

struct clusterLink {
	mstime_t ctime;
	int fd;
	sds sndbuf;
	sds rcvbuf;
	struct clusterNode *node;
};

typedef struct clusterLink clusterLink;

struct clusterNode {
	mstime_t ctime;
	char name[40];
	int flags;
	uint64_t configEpoch;
	unsigned char slots[2048];
	int numslots;
	int numslaves;
	struct clusterNode **slaves;
	struct clusterNode *slaveof;
	mstime_t ping_sent;
	mstime_t pong_received;
	mstime_t fail_time;
	mstime_t voted_time;
	mstime_t repl_offset_time;
	mstime_t orphaned_time;
	long long int repl_offset;
	char ip[46];
	int port;
	int cport;
	clusterLink *link;
	list *fail_reports;
};

enum __rlimit_resource {
	RLIMIT_CPU = 0,
	RLIMIT_FSIZE = 1,
	RLIMIT_DATA = 2,
	RLIMIT_STACK = 3,
	RLIMIT_CORE = 4,
	__RLIMIT_RSS = 5,
	RLIMIT_NOFILE = 7,
	__RLIMIT_OFILE = 7,
	RLIMIT_AS = 9,
	__RLIMIT_NPROC = 6,
	__RLIMIT_MEMLOCK = 8,
	__RLIMIT_LOCKS = 10,
	__RLIMIT_SIGPENDING = 11,
	__RLIMIT_MSGQUEUE = 12,
	__RLIMIT_NICE = 13,
	__RLIMIT_RTPRIO = 14,
	__RLIMIT_RTTIME = 15,
	__RLIMIT_NLIMITS = 16,
	__RLIM_NLIMITS = 16,
};

typedef __rlim64_t rlim_t;

enum __rusage_who {
	RUSAGE_SELF = 0,
	RUSAGE_CHILDREN = 4294967295,
	RUSAGE_THREAD = 1,
};

typedef enum __rlimit_resource __rlimit_resource_t;

typedef enum __rusage_who __rusage_who_t;

struct utsname {
	char sysname[65];
	char nodename[65];
	char release[65];
	char version[65];
	char machine[65];
	char domainname[65];
};

enum {
	_ISupper = 256,
	_ISlower = 512,
	_ISalpha = 1024,
	_ISdigit = 2048,
	_ISxdigit = 4096,
	_ISspace = 8192,
	_ISprint = 16384,
	_ISgraph = 32768,
	_ISblank = 1,
	_IScntrl = 2,
	_ISpunct = 4,
	_ISalnum = 8,
};

enum {
	PTHREAD_MUTEX_TIMED_NP = 0,
	PTHREAD_MUTEX_RECURSIVE_NP = 1,
	PTHREAD_MUTEX_ERRORCHECK_NP = 2,
	PTHREAD_MUTEX_ADAPTIVE_NP = 3,
};

enum {
	_SC_ARG_MAX = 0,
	_SC_CHILD_MAX = 1,
	_SC_CLK_TCK = 2,
	_SC_NGROUPS_MAX = 3,
	_SC_OPEN_MAX = 4,
	_SC_STREAM_MAX = 5,
	_SC_TZNAME_MAX = 6,
	_SC_JOB_CONTROL = 7,
	_SC_SAVED_IDS = 8,
	_SC_REALTIME_SIGNALS = 9,
	_SC_PRIORITY_SCHEDULING = 10,
	_SC_TIMERS = 11,
	_SC_ASYNCHRONOUS_IO = 12,
	_SC_PRIORITIZED_IO = 13,
	_SC_SYNCHRONIZED_IO = 14,
	_SC_FSYNC = 15,
	_SC_MAPPED_FILES = 16,
	_SC_MEMLOCK = 17,
	_SC_MEMLOCK_RANGE = 18,
	_SC_MEMORY_PROTECTION = 19,
	_SC_MESSAGE_PASSING = 20,
	_SC_SEMAPHORES = 21,
	_SC_SHARED_MEMORY_OBJECTS = 22,
	_SC_AIO_LISTIO_MAX = 23,
	_SC_AIO_MAX = 24,
	_SC_AIO_PRIO_DELTA_MAX = 25,
	_SC_DELAYTIMER_MAX = 26,
	_SC_MQ_OPEN_MAX = 27,
	_SC_MQ_PRIO_MAX = 28,
	_SC_VERSION = 29,
	_SC_PAGESIZE = 30,
	_SC_RTSIG_MAX = 31,
	_SC_SEM_NSEMS_MAX = 32,
	_SC_SEM_VALUE_MAX = 33,
	_SC_SIGQUEUE_MAX = 34,
	_SC_TIMER_MAX = 35,
	_SC_BC_BASE_MAX = 36,
	_SC_BC_DIM_MAX = 37,
	_SC_BC_SCALE_MAX = 38,
	_SC_BC_STRING_MAX = 39,
	_SC_COLL_WEIGHTS_MAX = 40,
	_SC_EQUIV_CLASS_MAX = 41,
	_SC_EXPR_NEST_MAX = 42,
	_SC_LINE_MAX = 43,
	_SC_RE_DUP_MAX = 44,
	_SC_CHARCLASS_NAME_MAX = 45,
	_SC_2_VERSION = 46,
	_SC_2_C_BIND = 47,
	_SC_2_C_DEV = 48,
	_SC_2_FORT_DEV = 49,
	_SC_2_FORT_RUN = 50,
	_SC_2_SW_DEV = 51,
	_SC_2_LOCALEDEF = 52,
	_SC_PII = 53,
	_SC_PII_XTI = 54,
	_SC_PII_SOCKET = 55,
	_SC_PII_INTERNET = 56,
	_SC_PII_OSI = 57,
	_SC_POLL = 58,
	_SC_SELECT = 59,
	_SC_UIO_MAXIOV = 60,
	_SC_IOV_MAX = 60,
	_SC_PII_INTERNET_STREAM = 61,
	_SC_PII_INTERNET_DGRAM = 62,
	_SC_PII_OSI_COTS = 63,
	_SC_PII_OSI_CLTS = 64,
	_SC_PII_OSI_M = 65,
	_SC_T_IOV_MAX = 66,
	_SC_THREADS = 67,
	_SC_THREAD_SAFE_FUNCTIONS = 68,
	_SC_GETGR_R_SIZE_MAX = 69,
	_SC_GETPW_R_SIZE_MAX = 70,
	_SC_LOGIN_NAME_MAX = 71,
	_SC_TTY_NAME_MAX = 72,
	_SC_THREAD_DESTRUCTOR_ITERATIONS = 73,
	_SC_THREAD_KEYS_MAX = 74,
	_SC_THREAD_STACK_MIN = 75,
	_SC_THREAD_THREADS_MAX = 76,
	_SC_THREAD_ATTR_STACKADDR = 77,
	_SC_THREAD_ATTR_STACKSIZE = 78,
	_SC_THREAD_PRIORITY_SCHEDULING = 79,
	_SC_THREAD_PRIO_INHERIT = 80,
	_SC_THREAD_PRIO_PROTECT = 81,
	_SC_THREAD_PROCESS_SHARED = 82,
	_SC_NPROCESSORS_CONF = 83,
	_SC_NPROCESSORS_ONLN = 84,
	_SC_PHYS_PAGES = 85,
	_SC_AVPHYS_PAGES = 86,
	_SC_ATEXIT_MAX = 87,
	_SC_PASS_MAX = 88,
	_SC_XOPEN_VERSION = 89,
	_SC_XOPEN_XCU_VERSION = 90,
	_SC_XOPEN_UNIX = 91,
	_SC_XOPEN_CRYPT = 92,
	_SC_XOPEN_ENH_I18N = 93,
	_SC_XOPEN_SHM = 94,
	_SC_2_CHAR_TERM = 95,
	_SC_2_C_VERSION = 96,
	_SC_2_UPE = 97,
	_SC_XOPEN_XPG2 = 98,
	_SC_XOPEN_XPG3 = 99,
	_SC_XOPEN_XPG4 = 100,
	_SC_CHAR_BIT = 101,
	_SC_CHAR_MAX = 102,
	_SC_CHAR_MIN = 103,
	_SC_INT_MAX = 104,
	_SC_INT_MIN = 105,
	_SC_LONG_BIT = 106,
	_SC_WORD_BIT = 107,
	_SC_MB_LEN_MAX = 108,
	_SC_NZERO = 109,
	_SC_SSIZE_MAX = 110,
	_SC_SCHAR_MAX = 111,
	_SC_SCHAR_MIN = 112,
	_SC_SHRT_MAX = 113,
	_SC_SHRT_MIN = 114,
	_SC_UCHAR_MAX = 115,
	_SC_UINT_MAX = 116,
	_SC_ULONG_MAX = 117,
	_SC_USHRT_MAX = 118,
	_SC_NL_ARGMAX = 119,
	_SC_NL_LANGMAX = 120,
	_SC_NL_MSGMAX = 121,
	_SC_NL_NMAX = 122,
	_SC_NL_SETMAX = 123,
	_SC_NL_TEXTMAX = 124,
	_SC_XBS5_ILP32_OFF32 = 125,
	_SC_XBS5_ILP32_OFFBIG = 126,
	_SC_XBS5_LP64_OFF64 = 127,
	_SC_XBS5_LPBIG_OFFBIG = 128,
	_SC_XOPEN_LEGACY = 129,
	_SC_XOPEN_REALTIME = 130,
	_SC_XOPEN_REALTIME_THREADS = 131,
	_SC_ADVISORY_INFO = 132,
	_SC_BARRIERS = 133,
	_SC_BASE = 134,
	_SC_C_LANG_SUPPORT = 135,
	_SC_C_LANG_SUPPORT_R = 136,
	_SC_CLOCK_SELECTION = 137,
	_SC_CPUTIME = 138,
	_SC_THREAD_CPUTIME = 139,
	_SC_DEVICE_IO = 140,
	_SC_DEVICE_SPECIFIC = 141,
	_SC_DEVICE_SPECIFIC_R = 142,
	_SC_FD_MGMT = 143,
	_SC_FIFO = 144,
	_SC_PIPE = 145,
	_SC_FILE_ATTRIBUTES = 146,
	_SC_FILE_LOCKING = 147,
	_SC_FILE_SYSTEM = 148,
	_SC_MONOTONIC_CLOCK = 149,
	_SC_MULTI_PROCESS = 150,
	_SC_SINGLE_PROCESS = 151,
	_SC_NETWORKING = 152,
	_SC_READER_WRITER_LOCKS = 153,
	_SC_SPIN_LOCKS = 154,
	_SC_REGEXP = 155,
	_SC_REGEX_VERSION = 156,
	_SC_SHELL = 157,
	_SC_SIGNALS = 158,
	_SC_SPAWN = 159,
	_SC_SPORADIC_SERVER = 160,
	_SC_THREAD_SPORADIC_SERVER = 161,
	_SC_SYSTEM_DATABASE = 162,
	_SC_SYSTEM_DATABASE_R = 163,
	_SC_TIMEOUTS = 164,
	_SC_TYPED_MEMORY_OBJECTS = 165,
	_SC_USER_GROUPS = 166,
	_SC_USER_GROUPS_R = 167,
	_SC_2_PBS = 168,
	_SC_2_PBS_ACCOUNTING = 169,
	_SC_2_PBS_LOCATE = 170,
	_SC_2_PBS_MESSAGE = 171,
	_SC_2_PBS_TRACK = 172,
	_SC_SYMLOOP_MAX = 173,
	_SC_STREAMS = 174,
	_SC_2_PBS_CHECKPOINT = 175,
	_SC_V6_ILP32_OFF32 = 176,
	_SC_V6_ILP32_OFFBIG = 177,
	_SC_V6_LP64_OFF64 = 178,
	_SC_V6_LPBIG_OFFBIG = 179,
	_SC_HOST_NAME_MAX = 180,
	_SC_TRACE = 181,
	_SC_TRACE_EVENT_FILTER = 182,
	_SC_TRACE_INHERIT = 183,
	_SC_TRACE_LOG = 184,
	_SC_LEVEL1_ICACHE_SIZE = 185,
	_SC_LEVEL1_ICACHE_ASSOC = 186,
	_SC_LEVEL1_ICACHE_LINESIZE = 187,
	_SC_LEVEL1_DCACHE_SIZE = 188,
	_SC_LEVEL1_DCACHE_ASSOC = 189,
	_SC_LEVEL1_DCACHE_LINESIZE = 190,
	_SC_LEVEL2_CACHE_SIZE = 191,
	_SC_LEVEL2_CACHE_ASSOC = 192,
	_SC_LEVEL2_CACHE_LINESIZE = 193,
	_SC_LEVEL3_CACHE_SIZE = 194,
	_SC_LEVEL3_CACHE_ASSOC = 195,
	_SC_LEVEL3_CACHE_LINESIZE = 196,
	_SC_LEVEL4_CACHE_SIZE = 197,
	_SC_LEVEL4_CACHE_ASSOC = 198,
	_SC_LEVEL4_CACHE_LINESIZE = 199,
	_SC_IPV6 = 235,
	_SC_RAW_SOCKETS = 236,
	_SC_V7_ILP32_OFF32 = 237,
	_SC_V7_ILP32_OFFBIG = 238,
	_SC_V7_LP64_OFF64 = 239,
	_SC_V7_LPBIG_OFFBIG = 240,
	_SC_SS_REPL_MAX = 241,
	_SC_TRACE_EVENT_NAME_MAX = 242,
	_SC_TRACE_NAME_MAX = 243,
	_SC_TRACE_SYS_MAX = 244,
	_SC_TRACE_USER_EVENT_MAX = 245,
	_SC_XOPEN_STREAMS = 246,
	_SC_THREAD_ROBUST_PRIO_INHERIT = 247,
	_SC_THREAD_ROBUST_PRIO_PROTECT = 248,
	_SC_MINSIGSTKSZ = 249,
	_SC_SIGSTKSZ = 250,
};

typedef unsigned char u8;

typedef unsigned int LZF_HSLOT;

typedef LZF_HSLOT LZF_STATE[65536];

typedef short unsigned int u16;

typedef struct {
	uint32_t state[5];
	uint32_t count[2];
	unsigned char buffer[64];
} SHA1_CTX;

typedef union {
	unsigned char c[64];
	uint32_t l[16];
} CHAR64LONG16;

typedef signed char __int8_t;

typedef short int __int16_t;

typedef __int8_t int8_t;

typedef __int16_t int16_t;

typedef __int32_t int32_t;

struct zlentry {
	unsigned int prevrawlensize;
	unsigned int prevrawlen;
	unsigned int lensize;
	unsigned int len;
	unsigned int headersize;
	unsigned char encoding;
	unsigned char *p;
};

typedef struct zlentry zlentry;

struct sockaddr_un;

struct _rio {
	size_t (*read)(struct _rio *, void *, size_t);
	size_t (*write)(struct _rio *, const void *, size_t);
	off_t (*tell)(struct _rio *);
	int (*flush)(struct _rio *);
	void (*update_cksum)(struct _rio *, const void *, size_t);
	uint64_t cksum;
	size_t processed_bytes;
	size_t max_processing_chunk;
	union {
		struct {
			sds ptr;
			off_t pos;
		} buffer;
		struct {
			FILE *fp;
			off_t buffered;
			off_t autosync;
		} file;
		struct {
			int *fds;
			int *state;
			int numfds;
			off_t pos;
			sds buf;
		} fdset;
	} io;
};

typedef struct _rio rio;

struct intset {
	uint32_t encoding;
	uint32_t length;
	int8_t contents[0];
};

typedef struct intset intset;

struct raxStack {
	void **stack;
	size_t items;
	size_t maxitems;
	void *static_items[32];
	int oom;
};

typedef struct raxStack raxStack;

struct raxIterator {
	int flags;
	rax *rt;
	unsigned char *key;
	void *data;
	size_t key_len;
	size_t key_max;
	unsigned char key_static_string[128];
	raxNode *node;
	raxStack stack;
};

typedef struct raxIterator raxIterator;

struct RedisModuleIO;

typedef void * (*moduleTypeLoadFunc)(struct RedisModuleIO *, int);

struct RedisModuleType;

typedef struct RedisModuleType moduleType;

struct RedisModuleCtx;

struct RedisModuleIO {
	size_t bytes;
	rio *rio;
	moduleType *type;
	int error;
	int ver;
	struct RedisModuleCtx *ctx;
};

typedef void (*moduleTypeSaveFunc)(struct RedisModuleIO *, void *);

typedef void (*moduleTypeRewriteFunc)(struct RedisModuleIO *, struct redisObject *, void *);

struct RedisModuleDigest;

typedef void (*moduleTypeDigestFunc)(struct RedisModuleDigest *, void *);

struct RedisModuleDigest {
	unsigned char o[20];
	unsigned char x[20];
};

typedef size_t (*moduleTypeMemUsageFunc)(const void *);

typedef void (*moduleTypeFreeFunc)(void *);

struct RedisModule;

struct RedisModuleType {
	uint64_t id;
	struct RedisModule *module;
	moduleTypeLoadFunc rdb_load;
	moduleTypeSaveFunc rdb_save;
	moduleTypeRewriteFunc aof_rewrite;
	moduleTypeMemUsageFunc mem_usage;
	moduleTypeDigestFunc digest;
	moduleTypeFreeFunc free;
	char name[10];
};

struct moduleValue {
	moduleType *type;
	void *value;
};

typedef struct moduleValue moduleValue;

struct zskiplistNode;

struct zskiplistLevel {
	struct zskiplistNode *forward;
	unsigned int span;
};

struct zskiplistNode {
	sds ele;
	double score;
	struct zskiplistNode *backward;
	struct zskiplistLevel level[0];
};

typedef struct zskiplistNode zskiplistNode;

struct zskiplist {
	struct zskiplistNode *header;
	struct zskiplistNode *tail;
	long unsigned int length;
	int level;
};

typedef struct zskiplist zskiplist;

struct zset {
	dict *dict;
	zskiplist *zsl;
};

typedef struct zset zset;

struct streamID {
	uint64_t ms;
	uint64_t seq;
};

typedef struct streamID streamID;

struct stream {
	rax *rax;
	uint64_t length;
	streamID last_id;
	rax *cgroups;
};

typedef struct stream stream;

struct streamCG {
	streamID last_id;
	rax *pel;
	rax *consumers;
};

typedef struct streamCG streamCG;

struct streamConsumer {
	mstime_t seen_time;
	sds name;
	rax *pel;
};

typedef struct streamConsumer streamConsumer;

typedef long unsigned int __dev_t;

typedef unsigned int __gid_t;

typedef long unsigned int __ino_t;

typedef long unsigned int __nlink_t;

typedef long int __blksize_t;

typedef long int __blkcnt_t;

struct timespec {
	__time_t tv_sec;
	__syscall_slong_t tv_nsec;
};

typedef struct RedisModuleIO RedisModuleIO;

struct streamNACK {
	mstime_t delivery_time;
	uint64_t delivery_count;
	streamConsumer *consumer;
};

typedef struct streamNACK streamNACK;

typedef struct {
	robj *subject;
	unsigned char encoding;
	unsigned char direction;
	quicklistIter *iter;
} listTypeIterator;

typedef struct {
	listTypeIterator *li;
	quicklistEntry entry;
} listTypeEntry;

typedef int (*__compar_fn_t)(const void *, const void *);

typedef struct {
	robj *subject;
	int encoding;
	int ii;
	dictIterator *di;
} setTypeIterator;

typedef struct {
	double min;
	double max;
	int minex;
	int maxex;
} zrangespec;

typedef struct {
	sds min;
	sds max;
	int minex;
	int maxex;
} zlexrangespec;

union _iterset {
	struct {
		intset *is;
		int ii;
	} is;
	struct {
		dict *dict;
		dictIterator *di;
		dictEntry *de;
	} ht;
};

union _iterzset {
	struct {
		unsigned char *zl;
		unsigned char *eptr;
		unsigned char *sptr;
	} zl;
	struct {
		zset *zs;
		zskiplistNode *node;
	} sl;
};

typedef struct {
	robj *subject;
	int type;
	int encoding;
	double weight;
	union {
		union _iterset set;
		union _iterzset zset;
	} iter;
} zsetopsrc;

typedef struct {
	int flags;
	unsigned char _buf[32];
	sds ele;
	unsigned char *estr;
	unsigned int elen;
	long long int ell;
	double score;
} zsetopval;

typedef union _iterset iterset;

typedef union _iterzset iterzset;

typedef struct {
	robj *subject;
	int encoding;
	unsigned char *fptr;
	unsigned char *vptr;
	dictIterator *di;
	dictEntry *de;
} hashTypeIterator;

struct moduleLoadQueueEntry {
	sds path;
	int argc;
	robj **argv;
};

struct configEnum {
	const char *name;
	const int val;
};

typedef struct configEnum configEnum;

struct rewriteConfigState {
	dict *option_to_line;
	dict *rewritten;
	int numlines;
	sds *lines;
	int has_tail;
};

struct streamIterator {
	stream *stream;
	streamID master_id;
	uint64_t master_fields_count;
	unsigned char *master_fields_start;
	unsigned char *master_fields_ptr;
	int entry_flags;
	int rev;
	uint64_t start_key[2];
	uint64_t end_key[2];
	raxIterator ri;
	unsigned char *lp;
	unsigned char *lp_ele;
	unsigned char *lp_flags;
	unsigned char field_buf[21];
	unsigned char value_buf[21];
};

typedef struct streamIterator streamIterator;

struct aofrwblock {
	long unsigned int used;
	long unsigned int free;
	char buf[10485760];
};

typedef struct aofrwblock aofrwblock;

struct pubsubPattern {
	client *client;
	robj *pattern;
};

typedef struct pubsubPattern pubsubPattern;

struct watchedKey {
	robj *key;
	redisDb *db;
};

typedef struct watchedKey watchedKey;

typedef long long int greg_t;

typedef greg_t gregset_t[23];

struct _libc_fpxreg {
	short unsigned int significand[4];
	short unsigned int exponent;
	short unsigned int __glibc_reserved1[3];
};

struct _libc_xmmreg {
	__uint32_t element[4];
};

struct _libc_fpstate {
	__uint16_t cwd;
	__uint16_t swd;
	__uint16_t ftw;
	__uint16_t fop;
	__uint64_t rip;
	__uint64_t rdp;
	__uint32_t mxcsr;
	__uint32_t mxcr_mask;
	struct _libc_fpxreg _st[8];
	struct _libc_xmmreg _xmm[16];
	__uint32_t __glibc_reserved1[24];
};

typedef struct _libc_fpstate *fpregset_t;

typedef struct {
	gregset_t gregs;
	fpregset_t fpregs;
	long long unsigned int __reserved1[8];
} mcontext_t;

struct ucontext_t {
	long unsigned int uc_flags;
	struct ucontext_t *uc_link;
	stack_t uc_stack;
	mcontext_t uc_mcontext;
	sigset_t uc_sigmask;
	struct _libc_fpstate __fpregs_mem;
	long long unsigned int __ssp[4];
};

typedef struct ucontext_t ucontext_t;

typedef struct RedisModuleDigest RedisModuleDigest;

typedef struct {
	const char *dli_fname;
	void *dli_fbase;
	const char *dli_sname;
	void *dli_saddr;
} Dl_info;

enum __itimer_which {
	ITIMER_REAL = 0,
	ITIMER_VIRTUAL = 1,
	ITIMER_PROF = 2,
};

struct itimerval {
	struct timeval it_interval;
	struct timeval it_value;
};

typedef enum __itimer_which __itimer_which_t;

struct _redisSortObject {
	robj *obj;
	union {
		double score;
		robj *cmpobj;
	} u;
};

typedef struct _redisSortObject redisSortObject;

struct _redisSortOperation {
	int type;
	robj *pattern;
};

typedef struct _redisSortOperation redisSortOperation;

struct clusterNodeFailReport {
	struct clusterNode *node;
	mstime_t time;
};

typedef struct clusterNodeFailReport clusterNodeFailReport;

typedef struct {
	char nodename[40];
	uint32_t ping_sent;
	uint32_t pong_received;
	char ip[46];
	uint16_t port;
	uint16_t cport;
	uint16_t flags;
	uint32_t notused1;
} clusterMsgDataGossip;

typedef struct {
	char nodename[40];
} clusterMsgDataFail;

typedef struct {
	uint32_t channel_len;
	uint32_t message_len;
	unsigned char bulk_data[8];
} clusterMsgDataPublish;

typedef struct {
	uint64_t configEpoch;
	char nodename[40];
	unsigned char slots[2048];
} clusterMsgDataUpdate;

typedef struct {
	uint64_t module_id;
	uint32_t len;
	uint8_t type;
	unsigned char bulk_data[3];
} clusterMsgModule;

union clusterMsgData {
	struct {
		clusterMsgDataGossip gossip[1];
	} ping;
	struct {
		clusterMsgDataFail about;
	} fail;
	struct {
		clusterMsgDataPublish msg;
	} publish;
	struct {
		clusterMsgDataUpdate nodecfg;
	} update;
	struct {
		clusterMsgModule msg;
	} module;
};

typedef struct {
	char sig[4];
	uint32_t totlen;
	uint16_t ver;
	uint16_t port;
	uint16_t type;
	uint16_t count;
	uint64_t currentEpoch;
	uint64_t configEpoch;
	uint64_t offset;
	char sender[40];
	unsigned char myslots[2048];
	char slaveof[40];
	char myip[46];
	char notused1[34];
	uint16_t cport;
	uint16_t flags;
	unsigned char state;
	unsigned char mflags[3];
	union clusterMsgData data;
} clusterMsg;

struct redisNodeFlags {
	uint16_t flag;
	char *name;
};

struct migrateCachedSocket {
	int fd;
	long int last_dbid;
	time_t last_use_time;
};

typedef struct migrateCachedSocket migrateCachedSocket;

struct slowlogEntry {
	robj **argv;
	int argc;
	long long int id;
	long long int duration;
	time_t time;
	sds cname;
	sds peerid;
};

typedef struct slowlogEntry slowlogEntry;

// typedef long int ptrdiff_t;

typedef int (*lua_CFunction)(lua_State *);

typedef double lua_Number;

typedef ptrdiff_t lua_Integer;

struct lua_Debug {
	int event;
	const char *name;
	const char *namewhat;
	const char *what;
	const char *source;
	int currentline;
	int nups;
	int linedefined;
	int lastlinedefined;
	char short_src[60];
	int i_ci;
};

typedef struct lua_Debug lua_Debug;

typedef void (*lua_Hook)(lua_State *, lua_Debug *);

struct ldbState {
	int fd;
	int active;
	int forked;
	list *logs;
	list *traces;
	list *children;
	int bp[64];
	int bpcount;
	int step;
	int luabp;
	sds *src;
	int lines;
	int currentline;
	sds cbuf;
	size_t maxlen;
	int maxlen_hint_sent;
};

typedef union {
	long long unsigned int __value64;
	struct {
		unsigned int __low;
		unsigned int __high;
	} __value32;
} __atomic_wide_counter;

struct __pthread_cond_s {
	__atomic_wide_counter __wseq;
	__atomic_wide_counter __g1_start;
	unsigned int __g_refs[2];
	unsigned int __g_size[2];
	unsigned int __g1_orig_size;
	unsigned int __wrefs;
	unsigned int __g_signals[2];
};

typedef long unsigned int pthread_t;

typedef union {
	char __size[4];
	int __align;
} pthread_condattr_t;

union pthread_attr_t {
	char __size[56];
	long int __align;
};

typedef union pthread_attr_t pthread_attr_t;

typedef union {
	struct __pthread_cond_s __data;
	char __size[48];
	long long int __align;
} pthread_cond_t;

enum {
	PTHREAD_CANCEL_ENABLE = 0,
	PTHREAD_CANCEL_DISABLE = 1,
};

enum {
	PTHREAD_CANCEL_DEFERRED = 0,
	PTHREAD_CANCEL_ASYNCHRONOUS = 1,
};

struct bio_job {
	time_t time;
	void *arg1;
	void *arg2;
	void *arg3;
};

struct bitfieldOp {
	uint64_t offset;
	int64_t i64;
	int opcode;
	int owtype;
	int bits;
	int sign;
};

struct redisReadTask {
	int type;
	int elements;
	int idx;
	void *obj;
	struct redisReadTask *parent;
	void *privdata;
};

typedef struct redisReadTask redisReadTask;

struct redisReplyObjectFunctions {
	void * (*createString)(const redisReadTask *, char *, size_t);
	void * (*createArray)(const redisReadTask *, int);
	void * (*createInteger)(const redisReadTask *, long long int);
	void * (*createNil)(const redisReadTask *);
	void (*freeObject)(void *);
};

typedef struct redisReplyObjectFunctions redisReplyObjectFunctions;

struct redisReader {
	int err;
	char errstr[128];
	char *buf;
	size_t pos;
	size_t len;
	size_t maxbuf;
	redisReadTask rstack[9];
	int ridx;
	void *reply;
	redisReplyObjectFunctions *fn;
	void *privdata;
};

typedef struct redisReader redisReader;

struct redisReply {
	int type;
	long long int integer;
	size_t len;
	char *str;
	size_t elements;
	struct redisReply **element;
};

typedef struct redisReply redisReply;

enum redisConnectionType {
	REDIS_CONN_TCP = 0,
	REDIS_CONN_UNIX = 1,
};

struct redisContext {
	int err;
	char errstr[128];
	int fd;
	int flags;
	char *obuf;
	redisReader *reader;
	enum redisConnectionType connection_type;
	struct timeval *timeout;
	struct {
		char *host;
		char *source_addr;
		int port;
	} tcp;
	struct {
		char *path;
	} unix_sock;
};

typedef struct redisContext redisContext;

struct redisAsyncContext;

typedef void redisCallbackFn(struct redisAsyncContext *, void *, void *);

typedef void redisDisconnectCallback(const struct redisAsyncContext *, int);

typedef void redisConnectCallback(const struct redisAsyncContext *, int);

struct redisCallback;

typedef struct redisCallback redisCallback;

struct redisCallbackList {
	redisCallback *head;
	redisCallback *tail;
};

typedef struct redisCallbackList redisCallbackList;

struct redisAsyncContext {
	redisContext c;
	int err;
	char *errstr;
	void *data;
	struct {
		void *data;
		void (*addRead)(void *);
		void (*delRead)(void *);
		void (*addWrite)(void *);
		void (*delWrite)(void *);
		void (*cleanup)(void *);
	} ev;
	redisDisconnectCallback *onDisconnect;
	redisConnectCallback *onConnect;
	redisCallbackList replies;
	struct {
		redisCallbackList invalid;
		struct dict *channels;
		struct dict *patterns;
	} sub;
};

struct redisCallback {
	struct redisCallback *next;
	redisCallbackFn *fn;
	void *privdata;
};

typedef struct redisAsyncContext redisAsyncContext;

struct sentinelAddr {
	char *ip;
	int port;
};

typedef struct sentinelAddr sentinelAddr;

struct instanceLink {
	int refcount;
	int disconnected;
	int pending_commands;
	redisAsyncContext *cc;
	redisAsyncContext *pc;
	mstime_t cc_conn_time;
	mstime_t pc_conn_time;
	mstime_t pc_last_activity;
	mstime_t last_avail_time;
	mstime_t act_ping_time;
	mstime_t last_ping_time;
	mstime_t last_pong_time;
	mstime_t last_reconn_time;
};

typedef struct instanceLink instanceLink;

struct sentinelRedisInstance {
	int flags;
	char *name;
	char *runid;
	uint64_t config_epoch;
	sentinelAddr *addr;
	instanceLink *link;
	mstime_t last_pub_time;
	mstime_t last_hello_time;
	mstime_t last_master_down_reply_time;
	mstime_t s_down_since_time;
	mstime_t o_down_since_time;
	mstime_t down_after_period;
	mstime_t info_refresh;
	int role_reported;
	mstime_t role_reported_time;
	mstime_t slave_conf_change_time;
	dict *sentinels;
	dict *slaves;
	unsigned int quorum;
	int parallel_syncs;
	char *auth_pass;
	mstime_t master_link_down_time;
	int slave_priority;
	mstime_t slave_reconf_sent_time;
	struct sentinelRedisInstance *master;
	char *slave_master_host;
	int slave_master_port;
	int slave_master_link_status;
	long long unsigned int slave_repl_offset;
	char *leader;
	uint64_t leader_epoch;
	uint64_t failover_epoch;
	int failover_state;
	mstime_t failover_state_change_time;
	mstime_t failover_start_time;
	mstime_t failover_timeout;
	mstime_t failover_delay_logged;
	struct sentinelRedisInstance *promoted_slave;
	char *notification_script;
	char *client_reconfig_script;
	sds info;
};

typedef struct sentinelRedisInstance sentinelRedisInstance;

struct sentinelState {
	char myid[41];
	uint64_t current_epoch;
	dict *masters;
	int tilt;
	int running_scripts;
	mstime_t tilt_start_time;
	mstime_t previous_time;
	list *scripts_queue;
	char *announce_ip;
	int announce_port;
	long unsigned int simfailure_flags;
};

struct sentinelScriptJob {
	int flags;
	int retry_num;
	char **argv;
	mstime_t start_time;
	pid_t pid;
};

typedef struct sentinelScriptJob sentinelScriptJob;

struct redisAeEvents {
	redisAsyncContext *context;
	aeEventLoop *loop;
	int fd;
	int reading;
	int writing;
};

typedef struct redisAeEvents redisAeEvents;

struct rewriteConfigState;

struct rusage;

struct readyList {
	redisDb *db;
	robj *key;
};

typedef struct readyList readyList;

struct sreamPropInfo {
	robj *keyname;
	robj *groupname;
};

typedef struct sreamPropInfo streamPropInfo;

struct hllhdr {
	char magic[4];
	uint8_t encoding;
	uint8_t notused[3];
	uint8_t card[8];
	uint8_t registers[0];
};

struct latencySample {
	int32_t time;
	uint32_t latency;
};

struct latencyTimeSeries {
	int idx;
	uint32_t max;
	struct latencySample samples[160];
};

struct latencyStats {
	uint32_t all_time_high;
	uint32_t avg;
	uint32_t min;
	uint32_t max;
	uint32_t mad;
	uint32_t samples;
	time_t period;
};

struct sequence {
	int length;
	int labels;
	struct sample *samples;
	double min;
	double max;
};

struct geoPoint {
	double longitude;
	double latitude;
	double dist;
	double score;
	char *member;
};

typedef struct geoPoint geoPoint;

struct geoArray {
	struct geoPoint *array;
	size_t buckets;
	size_t used;
};

typedef struct geoArray geoArray;

typedef struct {
	uint64_t bits;
	uint8_t step;
} GeoHashBits;

typedef struct {
	double min;
	double max;
} GeoHashRange;

typedef struct {
	GeoHashBits hash;
	GeoHashRange longitude;
	GeoHashRange latitude;
} GeoHashArea;

typedef struct {
	GeoHashBits north;
	GeoHashBits east;
	GeoHashBits west;
	GeoHashBits south;
	GeoHashBits north_east;
	GeoHashBits south_east;
	GeoHashBits north_west;
	GeoHashBits south_west;
} GeoHashNeighbors;

typedef uint64_t GeoHashFix52Bits;

typedef struct {
	GeoHashBits hash;
	GeoHashArea area;
	GeoHashNeighbors neighbors;
} GeoHashRadius;

enum {
	PTHREAD_MUTEX_TIMED_NP___2 = 0,
	PTHREAD_MUTEX_RECURSIVE_NP___2 = 1,
	PTHREAD_MUTEX_ERRORCHECK_NP___2 = 2,
	PTHREAD_MUTEX_ADAPTIVE_NP___2 = 3,
	PTHREAD_MUTEX_NORMAL = 0,
	PTHREAD_MUTEX_RECURSIVE = 1,
	PTHREAD_MUTEX_ERRORCHECK = 2,
	PTHREAD_MUTEX_DEFAULT = 0,
	PTHREAD_MUTEX_FAST_NP = 0,
};

struct RedisModule {
	void *handle;
	char *name;
	int ver;
	int apiver;
	list *types;
};

struct RedisModuleBlockedClient;

struct AutoMemEntry;

struct RedisModulePoolAllocBlock;

struct RedisModuleCtx {
	void *getapifuncptr;
	struct RedisModule *module;
	client *client;
	struct RedisModuleBlockedClient *blocked_client;
	struct AutoMemEntry *amqueue;
	int amqueue_len;
	int amqueue_used;
	int flags;
	void **postponed_arrays;
	int postponed_arrays_count;
	void *blocked_privdata;
	int *keys_pos;
	int keys_count;
	struct RedisModulePoolAllocBlock *pa_head;
};

typedef uint64_t RedisModuleTimerID;

typedef struct RedisModule RedisModule;

struct AutoMemEntry {
	void *ptr;
	int type;
};

struct RedisModulePoolAllocBlock {
	uint32_t size;
	uint32_t used;
	struct RedisModulePoolAllocBlock *next;
	char memory[0];
};

typedef struct RedisModulePoolAllocBlock RedisModulePoolAllocBlock;

typedef struct RedisModuleCtx RedisModuleCtx;

typedef int (*RedisModuleCmdFunc)(RedisModuleCtx *, void **, int);

typedef void (*RedisModuleDisconnectFunc)(RedisModuleCtx *, struct RedisModuleBlockedClient *);

struct RedisModuleBlockedClient {
	client *client;
	RedisModule *module;
	RedisModuleCmdFunc reply_callback;
	RedisModuleCmdFunc timeout_callback;
	RedisModuleDisconnectFunc disconnect_callback;
	void (*free_privdata)(RedisModuleCtx *, void *);
	void *privdata;
	client *reply_client;
	int dbid;
};

struct RedisModuleKey {
	RedisModuleCtx *ctx;
	redisDb *db;
	robj *key;
	robj *value;
	void *iter;
	int mode;
	uint32_t ztype;
	zrangespec zrs;
	zlexrangespec zlrs;
	uint32_t zstart;
	uint32_t zend;
	void *zcurrent;
	int zer;
};

typedef struct RedisModuleKey RedisModuleKey;

struct RedisModuleCommandProxy {
	struct RedisModule *module;
	RedisModuleCmdFunc func;
	struct redisCommand *rediscmd;
};

typedef struct RedisModuleCommandProxy RedisModuleCommandProxy;

struct RedisModuleCallReply {
	RedisModuleCtx *ctx;
	int type;
	int flags;
	size_t len;
	char *proto;
	size_t protolen;
	union {
		const char *str;
		long long int ll;
		struct RedisModuleCallReply *array;
	} val;
};

typedef struct RedisModuleCallReply RedisModuleCallReply;

typedef struct RedisModuleBlockedClient RedisModuleBlockedClient;

typedef int (*RedisModuleNotificationFunc)(RedisModuleCtx *, int, const char *, robj *);

struct RedisModuleKeyspaceSubscriber {
	RedisModule *module;
	RedisModuleNotificationFunc notify_callback;
	int event_mask;
	int active;
};

typedef struct RedisModuleKeyspaceSubscriber RedisModuleKeyspaceSubscriber;

typedef void (*RedisModuleClusterMessageReceiver)(RedisModuleCtx *, const char *, uint8_t, const unsigned char *, uint32_t);

struct moduleClusterReceiver {
	uint64_t module_id;
	RedisModuleClusterMessageReceiver callback;
	struct RedisModule *module;
	struct moduleClusterReceiver *next;
};

typedef struct moduleClusterReceiver moduleClusterReceiver;

typedef void (*RedisModuleTimerProc)(RedisModuleCtx *, void *);

struct RedisModuleTimer {
	RedisModule *module;
	RedisModuleTimerProc callback;
	void *data;
};

typedef struct RedisModuleTimer RedisModuleTimer;

struct typemethods {
	uint64_t version;
	moduleTypeLoadFunc rdb_load;
	moduleTypeSaveFunc rdb_save;
	moduleTypeRewriteFunc aof_rewrite;
	moduleTypeMemUsageFunc mem_usage;
	moduleTypeDigestFunc digest;
	moduleTypeFreeFunc free;
};

struct evictionPoolEntry {
	long long unsigned int idle;
	sds key;
	sds cached;
	int dbid;
};

struct redisAsyncContext___2;

typedef void redisCallbackFn___2(struct redisAsyncContext___2 *, void *, void *);

typedef void redisDisconnectCallback___2(const struct redisAsyncContext___2 *, int);

typedef void redisConnectCallback___2(const struct redisAsyncContext___2 *, int);

struct redisCallback___2;

typedef struct redisCallback___2 redisCallback___2;

struct redisCallbackList___2 {
	redisCallback___2 *head;
	redisCallback___2 *tail;
};

typedef struct redisCallbackList___2 redisCallbackList___2;

struct dict___2;

struct redisAsyncContext___2 {
	redisContext c;
	int err;
	char *errstr;
	void *data;
	struct {
		void *data;
		void (*addRead)(void *);
		void (*delRead)(void *);
		void (*addWrite)(void *);
		void (*delWrite)(void *);
		void (*cleanup)(void *);
	} ev;
	redisDisconnectCallback___2 *onDisconnect;
	redisConnectCallback___2 *onConnect;
	redisCallbackList___2 replies;
	struct {
		redisCallbackList___2 invalid;
		struct dict___2 *channels;
		struct dict___2 *patterns;
	} sub;
};

struct redisCallback___2 {
	struct redisCallback___2 *next;
	redisCallbackFn___2 *fn;
	void *privdata;
};

struct dictEntry___2;

typedef struct dictEntry___2 dictEntry___2;

struct dictType___2;

typedef struct dictType___2 dictType___2;

struct dict___2 {
	dictEntry___2 **table;
	dictType___2 *type;
	long unsigned int size;
	long unsigned int sizemask;
	long unsigned int used;
	void *privdata;
};

typedef struct redisAsyncContext___2 redisAsyncContext___2;

struct dictEntry___2 {
	void *key;
	void *val;
	struct dictEntry___2 *next;
};

struct dictType___2 {
	unsigned int (*hashFunction)(const void *);
	void * (*keyDup)(void *, const void *);
	void * (*valDup)(void *, const void *);
	int (*keyCompare)(void *, const void *, const void *);
	void (*keyDestructor)(void *, void *);
	void (*valDestructor)(void *, void *);
};

typedef struct dict___2 dict___2;

struct dictIterator___2 {
	dict___2 *ht;
	int index;
	dictEntry___2 *entry;
	dictEntry___2 *nextEntry;
};

typedef struct dictIterator___2 dictIterator___2;

#ifndef BPF_NO_PRESERVE_ACCESS_INDEX
#pragma clang attribute pop
#endif

#endif /* __VMLINUX_H__ */
