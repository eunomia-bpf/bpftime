# Maintainer decisions

Design and process questions the maintainers have settled, newest first. Each
entry gives the date, the decision, the pull requests or issues it came from,
and the maintainer's words verbatim.

How this list is used:

- Read it before starting a change. Authors and reviewers check every pull
  request against it.
- A pull request that reverses an entry is a design problem until a
  maintainer decides again.
- A new or changed decision is added here, with its date, its pull request and
  the maintainer's words verbatim, before or in the change that relies on it.
  An entry is never edited to say something its maintainer did not say.
  Superseding it means adding a new entry that names the old one.

## 2026-11-26: Record maintainer decisions; agree on new mechanisms before writing them

**Decision.**

1. Maintainer decisions are recorded in this file, with dates and links.
2. A new lock or other synchronization primitive, a new shared-memory layout
   or attach path, a new thread, constructor, environment variable or
   protocol, or a change of more than about 100 production lines, needs a
   maintainer's agreement before it is written. The agreement can be in an
   issue or in a maintainer's comment on the pull request, and the pull
   request links it.
3. A review that suggests a different design writes it as an option for the
   maintainers to choose, not as a fix the author must make.

**From:** #948, #950.

**Maintainer's words:**

> “1. 在 CLAUDE.md 里记录维护者决定： 列一个带日期和 PR 链接的"Decisions"清单。推翻其中某条的 PR，在维护者重新决定之前一律算 design problem。”

> “2. 新机制先开 issue 谈好再写： 新的锁、共享内存布局、线程、构造函数、环境变量或协议，或者生产代码超过约 100 行，都先在 issue 里讨论。review 里提的替代设计标成"供维护者选择"，而不是要求作者修。”

> “新机制没有附上维护者同意过的 issue，给 discussion。我自己在 comment 里面评论也行, 不一定是 issue.”

> “Maintainer decisions 单独一个文件是不是更好? 你看看? 要记录原话”

## 2026-11-26: The `LD_PRELOAD` agent runs no background thread and never blocks or waits

**Decision.** The injected agent and its shim start no background thread and
never block or wait. Work that needs a thread or a wait belongs in the CLI or
the syscall server. This is why the auto-refresh thread was removed (#927) and
startup waits and blocking locks moved out of the agent into the CLI (#935).
The threads master still starts in or next to the agent are listed in #948 as
follow-ups.

**From:** #927, #935, #948.

**Maintainer's words:**

> “删掉 auto-refresh 线程是必须删掉的, 不能后台线程, 不能任何 block 和等待再 shim”

## 2026-11-25: Cross-process shared-memory locks are glibc robust mutexes

**Decision.** Locks shared between processes through the shared-memory segment
use `pthread_mutex_t` with `PTHREAD_PROCESS_SHARED | PTHREAD_MUTEX_ROBUST`, not
a self-written spinlock with liveness probes. Their measured cost is accepted:
about 13-15 ns more per uncontended lock than a spinlock, and 1.4-1.7x under
contention. In return, dead-holder recovery is guaranteed by the kernel's
robust futex list, across PID namespaces included. #940 later replaced this
with an owner-token spinlock; see #948.

**From:** #922, #923, #929.

**Maintainer's words:**

> “如果能接受每次多 13 ns 左右、竞争时慢 1.4–1.7 倍，就保留 glibc robust mutex，可靠性由内核保证。”

## 2026-11-25: Every pull request carries a benchmark and a performance impact analysis

**Decision.** Every pull request fills in the "Benchmark and performance
impact" section of the pull request template. The analysis covers the hot
paths it touches. The benchmark compares master and the change on the same
machine and build options, with a multi-thread or multi-process run when a
lock or shared state is touched.

**From:** #934.

**Maintainer's words:**

> “以后要求规定一下 bpftime 所有都要带 benchmark 和性能影响分析? 是不是在 PR template 里面说?”
