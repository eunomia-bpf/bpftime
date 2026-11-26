# Maintainer decisions

Newest first. Read before starting a change. A pull request that reverses an entry is a design problem until a maintainer decides again. Every entry quotes the maintainer verbatim; changing a decision means adding a new entry.

- **2026-11-26: safety principles for the `LD_PRELOAD` agent** (#948). In the injected agent (the syscall server is out of scope): only a limited set of fork-safe lock mechanisms; nothing that can throw an error into the host or make it stop; no background thread; no waiting or blocking; no signal mechanism.
  > 大概就是我觉得 LD PRELOAD 里面有一些安全原则最好要遵守, 比如说 lock 必须只有有限的, fork 安全的 lock 机制; 不要任何可能会抛出错误或者导致宿主停下来的情况; 不要任何后台线程, 不要任何等待或者 block; 不要任何信号机制 这些也要记录进去
  >
  > syscall server 没关系; 记录我的原话
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
