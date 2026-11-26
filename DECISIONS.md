# Maintainer decisions

Newest first. Read before starting a change. A pull request that reverses an entry is a design problem until a maintainer decides again. Every entry quotes the maintainer verbatim; changing a decision means adding a new entry.

- **2026-11-26: zero active syscalls on `LD_PRELOAD` hot paths ("Four Zeros"); revert `#906` ringbuf eventfd** (#906, #946). Injected `LD_PRELOAD` hot paths (uprobe/uretprobe/syscall callbacks, BPF helpers, ringbuf/map reserve/submit/lookup/update, constructor early-exit and process-exit paths) must issue **zero active syscalls** (`write`, `read`, `socket`, `connect`, `sendmsg`, `recvmsg`, `open`, `mmap`, `eventfd`, etc.), preserve host `errno`, and never hold `static` C++ heap containers whose `__cxa_atexit` destructors race background threads at `exit(0)`. Collector / `syscall-server` consumer wait paths are out of scope. Explicit maintainer-approved exemptions on hot paths are strictly limited to:
  1. Linux kernel `futex` inside `PTHREAD_MUTEX_ROBUST` only on actual cross-process contention or `EOWNERDEAD` dead-owner recovery (uncontended path is pure userspace atomic CAS);
  2. `process_vm_readv` / `process_vm_writev` when opt-in fault-checked `bpf_probe_read*` / `bpf_probe_write_user` (`ENABLE_PROBE_READ_CHECK`) is explicitly enabled;
  3. One-time cold-start per-thread / per-process identity cache initialization into TLS or `constinit` `.bss` (`SYS_gettid`, `readlink("/proc/self/exe")`), never on steady-state events;
  4. Userspace vDSO `clock_gettime` (`bpf_ktime_get_ns`).
  Any other hot-path syscall requires explicit prior maintainer approval recorded here. `#906` (`cb271dde`) and `#946` (`e368a0f2`) are reverted verbatim back to pure shared-memory ringbuf submission.
  > 我们是不是也要要求热路径没有 syscall? 之前 ring buf 不是 event fd 的时候好像没很多 bug?
  >
  > 我们现在崩溃是崩溃业务进程还是 collector? collector 问题不大
  >
  > 热路径零主动 syscall：除跨进程 PTHREAD_MUTEX_ROBUST 极少数发生真实锁竞争/死锁恢复时进入内核 futex 外，热路径严禁任何 write / read / socket / connect / recvmsg / open / mmap 系统调用（既防耗时与阻塞，也防 syscall_trace/seccomp 递归重入和沙箱拦截）。
  >
  > 开 PR reviert 906, 一字不差回复原本的, 确保原本的没 bug 并且能 work, 要真实测试 [workload]; 并且把我的这个话写进去, 除非特批的不然.
  >
  > 906 相关的别的 PR 可以关闭了. 顺便也看看我们还有没有别的一些, 比如说同一时间或者最近引入的类似的问题, 修改了原本能 work 的机制没有充分讨论并且违规
- **2026-11-26: safety rules for the `LD_PRELOAD` agent** (#948, #951). The syscall server is out of scope.
  - **Locks:** only lock-free atomics or fixed arrays, and glibc robust mutexes across processes. On hot paths take them with `trylock`; if the lock is held, skip the event and count it, never wait. (`lock()` blocks while a live holder runs, and forever if the holder is stopped by `SIGSTOP` or a debugger.)
  - **Errors:** every hook entry and `extern "C"` boundary catches all exceptions; injected code contains no `abort`, `exit`, `assert` or `std::terminate`.
  - **Threads:** a background thread exists only as an optional compile-time flag.
  - **Waiting:** bpftime's own code never sleeps, polls, joins, calls `waitpid` or takes a blocking lock. The only waits allowed are blocking calls the host itself makes, such as the syscall server's emulated `epoll_wait` or `read`, and those last only as long as the host asked.
  - **Signals:** no installing or replacing signal handlers, no sending signals, no changing signal masks; a `kill(pid, 0)` liveness probe does not count. Exception: the trap uprobe backend (`SIGTRAP`), used where Frida is unavailable.
  > 大概就是我觉得 LD PRELOAD 里面有一些安全原则最好要遵守, 比如说 lock 必须只有有限的, fork 安全的 lock 机制; 不要任何可能会抛出错误或者导致宿主停下来的情况; 不要任何后台线程, 不要任何等待或者 block; 不要任何信号机制 这些也要记录进去
  >
  > syscall server 没关系; 记录我的原话
  >
  > 锁： 只允许两类：
  >
  > * 无锁原子操作或固定数组；
  > * 跨进程用 glibc robust mutex（11-25 已决定）。关键一点：mutex 被别人持有时 lock() 本身就会阻塞，robust 只管持有者死掉的情况；持有者被 SIGSTOP 或调试器停住时，所有进程都会一直等。所以热路径上应该用 trylock，拿不到就跳过这次事件并计数，不等待。这样第 1 条和第 4 条才不打架。不让宿主出错或停下： 所有 hook 入口和 `extern "C"` 边界都要捕获所有异常；注入代码里不出现 `abort`、`exit`、`assert`、`std::terminate`。这条和 CLAUDE.md 现有的第 1 条不变量一致，只是说得更死。
  >
  > 后台线程可以在编译 flag 里面可选不等待、不阻塞： bpftime 自己不 sleep、不轮询、不 join、不 waitpid，也不用阻塞式的锁。唯一例外是宿主自己发起的阻塞调用：syscall server 模拟的 epoll_wait、read，只能按宿主要求的时长等，否则 syscall 模拟就坏了。不用信号： 不安装或替换信号处理函数，不发信号，不改信号掩码。用 `kill(pid, 0)` 只是探测进程是否存活，不算。
  >
  > * 冲突： 没有 Frida 的架构靠 trap 后端（SIGTRAP），这条一定下，trap 后端单独定为例外
- **2026-11-26: decisions are recorded here; new mechanisms are agreed on first** (#948, #950). A new lock, shared-memory layout or attach path, thread, constructor, environment variable or protocol, or a change of more than ~100 production lines, needs a maintainer's agreement before it is written, in an issue or in a maintainer's comment on the pull request. Alternative designs suggested in reviews are options for the maintainers, not fixes the author must make.
  > 1. 在 CLAUDE.md 里记录维护者决定： 列一个带日期和 PR 链接的"Decisions"清单。推翻其中某条的 PR，在维护者重新决定之前一律算 design problem。
  >
  > 2. 新机制先开 issue 谈好再写： 新的锁、共享内存布局、线程、构造函数、环境变量或协议，或者生产代码超过约 100 行，都先在 issue 里讨论。review 里提的替代设计标成"供维护者选择"，而不是要求作者修。
  >
  > 新机制没有附上维护者同意过的 issue，给 discussion。我自己在 comment 里面评论也行, 不一定是 issue.
  >
  > Maintainer decisions 单独一个文件是不是更好? 你看看? 要记录原话
- **2026-11-26: the `LD_PRELOAD` agent runs no background thread and never blocks or waits** (#927, #935).
  > 删掉 auto-refresh 线程是必须删掉的, 不能后台线程, 不能任何 block 和等待再 shim
- **2026-11-25: cross-process shared-memory locks are glibc robust mutexes** (`PTHREAD_PROCESS_SHARED | PTHREAD_MUTEX_ROBUST`; #922, #923, #929). The cost, about +13–15 ns per uncontended lock and 1.4–1.7x under contention, is accepted. #940 departs from this; see #948.
  > 如果能接受每次多 13 ns 左右、竞争时慢 1.4–1.7 倍，就保留 glibc robust mutex，可靠性由内核保证。
- **2026-11-25: every pull request carries a benchmark and a performance impact analysis** (#934).
  > 以后要求规定一下 bpftime 所有都要带 benchmark 和性能影响分析? 是不是在 PR template 里面说?
