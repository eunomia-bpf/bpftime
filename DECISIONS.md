# Maintainer decisions

Newest first. Read before starting a change. A pull request that reverses an entry is a design problem until a maintainer decides again. Every entry quotes the maintainer verbatim; changing a decision means adding a new entry.

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
