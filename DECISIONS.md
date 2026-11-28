# Maintainer decisions

Newest first. Read before starting a change. A pull request that reverses an entry is a design problem until a maintainer decides again. Every entry quotes the maintainer verbatim; changing a decision means adding a new entry.

- **2026-11-27: one shared-memory slot per CPU, leased lazily across the outermost `bpftime_prog_exec`, borrowing the next free slot when a preempted holder has it** (#1028; supersedes the 2026-11-27 `#1001` lock-free `PERCPU_ARRAY` entry below and registers `bpftime_vcpu_locks` in L1 of `robust_shm_mutex.hpp`).
  - **Problem.** Userspace cannot disable kernel preemption like `preempt_disable()`. An unguarded per-CPU slot lets a timer-preempted thread and the incoming thread on the same core clobber multi-tail-call `PERCPU_ARRAY` scratch state, such as 16 KB stack-unwinder records.
  - **Slots.** Instead of unconditional entry locks (`#730`/`#937`) or a global lock, `bpftime_shm` holds one `robust_shm_recovering_mutex` slot per possible CPU (`bpftime_vcpu_locks`).
  - **Lease.** `bpftime_prog_exec` still takes zero locks and makes zero `sched_getcpu()` calls on entry.
    - On the first `my_sched_getcpu()` inside an outermost exec (`PERCPU_ARRAY`, `PERCPU_HASH`, `bpf_get_smp_processor_id()`), the thread `try_lock`s its own CPU's slot.
    - If a preempted holder still has it, the thread borrows the next free slot after it, which belongs to another CPU.
    - It keeps the slot across all nested tail calls until `bpftime_leave_bpf_exec()`.
    - Only with every slot busy do that exec's per-CPU accesses fail, counted in `shm_trylock_skips`.
  - **Borrowing.** A borrowed slot's values are counted under the other CPU, and that CPU's own thread may then borrow the next slot. Sums over CPUs stay exact, and the slot stays exclusive, so no scratch is shared.
  - **Layout.** Per-CPU maps keep one column per possible CPU, each CPU's value rounded up to 8 bytes as in the bpf syscall layout. Tools read the real CPU lists.
  - **CPU ids.** Every CPU id a BPF program reads is its leased slot: per-CPU map columns, `bpf_get_smp_processor_id()` and the `BPF_F_CURRENT_CPU` perf ring. Perf output without a lease uses the real CPU.
  - **No slots.** Without slots (no global shm, or a segment from an older version), programs see the real CPU.
  - **Replaces.** This replaces the extra-slot design that an earlier revision of #1028 tried: `ceil(possible CPUs * BPFTIME_VCPU_RATIO) + 1` slots, with the `possible` and `online` lists mocked to match.
  > 这直接说明上游 PerCPURecord 从设计第一天起就要求整条多级尾调用回溯期间独占槽位、绝不允许并发覆盖；内核靠「preempt_disable + bpf_prog_active + 双表隔离中断」保证了这一点，而用户态一旦完全去掉槽位互斥，普通时钟中断抢占就会直接打破这个前提。那么用户态应该怎么做? 直接全局锁还是?
  >
  > 或者全局概念上说, 是不是我们有一个虚拟的 cpu, 数量和真实 cpu 相同(避免动态分配), 但是实际上会在每个入口 assign 到活跃的 cpu 上面? 能 match 的就 match, 那么和进程关系是什么? 具体是什么形态?
  >
  > 是通用的机制, 不是 adhoc 的吗? 以后别的 per cpu map 也是复用? 对于所有的都生效, [profiler] 能不能尽可能不要改?
  >
  > 该修掉的都修, 并且要测试, 尽可能最小修改, 生产代码增加不超过 50-100? 或者尽可能不要增加?
  >
  > 用真实 libbpf 工具端到端测一下 possible 文件, 也要用真实的 otel profiler 跑一下试试; 另外, 确保记录一下, 所有读取 cpu id 的都应该走这个 vcpu 吧?
  >
  > 是不是假如我们还是回到原先的, 没有多的槽位, 通过退避 + lock 的方式代码改动最小? 我们为啥不 spin lock 之类的? 说说? 很多问题都是改动 vcpu 引起的?
  >
  > 每个 CPU 一个槽位，不是拿不到可以试试下一个槽位吗
  >
  > 借槽：不用额外列，去借别的 CPU 空着的槽，也不丢。但数据会记到别的 CPU 名下，而且会把那个 CPU 自己的线程挤走，你之前否掉了。我觉得没问题
  >
  > 就是之前的版本, 我们是直接回到之前的 commit 还是?
- **2026-11-27: `HASH_OF_MAPS` stores its inner map ids in `bpftime_hash_map`** (supersedes `HASH_OF_MAPS` in the L2 `bpf_map_handler::map_lock` line of the lock inventory below). `hash_map_of_maps_impl` moves from `var_size_hash_map_impl` (a Boost `unordered_map` under `map_lock`) to the preallocated `bpftime_hash_map` that `HASH` uses, and sets `should_lock = false`: helper lookups take no lock, and the syscall-only updates and deletes take the recover-and-continue write stripes (#1020). The syscall lookup of both map-in-map types no longer takes `map_lock` either: array slots are never freed, and the hash variant copies the 4-byte inner id while `bpftime_hash_map::find_slot` still checks the slot generation, so a concurrent delete and slot reuse restarts the walk instead of returning another key's id. `bpftime_hash_map` writes and reads an aligned 4-byte value with one access, so a lock-free lookup never reads a half-written inner id.
  > HASH_OF_MAPS 建在 var_hash_map 上 value 只是 4 字节的 map id，改建在 bpftime_hash_map 上 小 这个先改动
  >
  > HASH_OF_MAPS 那个也去做, 做完去 review 一下别的部分

- **2026-11-27: `LPM_TRIE` lookups take no lock; writes take one recover-and-continue lock over a preallocated node pool** (supersedes `LPM_TRIE` in the L2 `bpf_map_handler::map_lock` line of the lock inventory below; replaces the 64 prefix shards of #1021). `lpm_trie_map_impl` keeps `should_lock = false`. At creation it allocates `2 * max_entries` fixed nodes (at most `max_entries` leaves and `max_entries - 1` intermediate nodes) linked by `uint32_t` indices; no Boost allocator runs after creation. Writes take one `robust_shm_recovering_mutex` (L1): `try_lock` on helper paths, which skips and counts on contention, and a blocking lock on syscall paths. As in kernel `lpm_trie.c`, an insert builds its nodes completely and links them with one store, and a delete unlinks with one store, so a writer that dies at any step leaves the tree intact; it leaks at most the two nodes it held and may leave the entry count one high. Freed nodes enter a FIFO ring and are recycled only after every node freed before them, so a value pointer returned by a lookup stays readable until the ring cycles. A lookup records the ring's free position before its walk and retries if allocation passed that position by the end, since only then can a node freed during the walk have been rewritten; after 8 attempts it returns NULL and counts in `shm_trylock_skips`. This is the before-and-after version check of point 3 below, using the ring positions as the version: they never stay odd after a writer dies, so no repair step is needed. An update of an existing key still writes the value in place.
  > LPM_TRIE 怎么做
  > 目标：查找完全不拿锁，因此任何争用都不会让查找失败；持有者死掉也不会坏，所以不需要毒化。
  >
  > 1. 节点放进预分配的池子里。创建 map 时一次性分配 `2*max_entries` 个定长节点（叶子不超过 `max_entries` 个，中间节点不超过 `max_entries - 1` 个）。每个节点存前缀长度、状态、两个子节点下标、前缀数据和 value。节点之间用 `uint32_t` 下标连接，不用 `offset_ptr`，运行时也不再调用 Boost 分配器。
  > 2. 写操作拿一把锁，用"恢复后继续"的规则。插入时先把新节点完整写好，再用一次写把它挂到父节点上；删除时用一次写摘下来。所以持有者死在任何一步，树都是完整的，最多漏一个节点，和 HASH 的情况一样。LPM 表的写很少（比如 profiler 只在 mmap/munmap 时写），一把锁就够，#1021 的 64 片分片也就用不上了。
  > 3. 读操作用 seqlock，不拿锁。写的人在修改前后各把版本号加一；读的人读之前和读之后各看一次版本号，变了就重读，重读几次还不行就返回 NULL。写的人如果死在中途，版本号会停在奇数，下一个写的人拿锁时发现上一个持有者死了，就把它修回偶数。
  > 4. 被删的节点不要马上复用。查找返回的是指向节点里 value 的指针，eBPF 程序可能在 helper 返回后还在读。内核靠 RCU 保证这个指针在读的期间有效；我们改成先进先出的空闲链表：一个节点被释放后，要再经过很多次释放才会被重新分配。这起到隔离期的作用，虽然不是 RCU 那样的硬保证，但实际够用；而且节点池一直映射着，就算读到复用中的节点，也只会读到旧数据，不会崩。
  > 5. 怎么验证：
  >    * 用 otel profiler 实测，对比 master 和改后"查找因为锁失败"的次数（`shm_trylock_skips`）以及查不到的次数；
  >    * 单元测试覆盖三种情况：多个线程查找的同时有线程不停插入和删除；写的人死在中途（用 #1020 那种 fork 加 SIGSEGV 的办法）；同一个 PID 的多线程查找。去做.

- **2026-11-27: `bpftime_hash_map` write stripes recover from a dead owner instead of poisoning** (#1020; supersedes the `bpftime_hash_map` stripes in the L2 line of the lock inventory below, which move to the L1 recover-and-continue policy). The `HASH`/`PERCPU_HASH` write path never leaves a bucket chain broken: an insert publishes the new slot with one release store to `head_idx`, an erase unlinks with one store, and the slot free list is lock-free. A writer that dies at any step therefore leaves every chain intact and leaks at most one slot, with `_count` one higher. The 64 stripes are `robust_shm_recovering_mutex`, a `robust_shm_mutex` whose `lock()`/`try_lock()` mark the mutex consistent on `EOWNERDEAD` and continue instead of latching; the call sites keep `with_robust_shm_lock` (`try_lock` on helper paths, which still skips and counts on contention, blocking `lock` on syscall paths), so a dead owner no longer leaves 1/64 of the map returning `EBUSY` for the session (the `pid_events` case after a child is killed). No per-step journal, no pool lock and no extra preallocated slots: a reserve cannot absorb the leak while the slot and `_count` drift together, and a leak only shows once the map is full. Maps built on Boost containers keep poison-on-death. `robust_shm_mutex::try_lock()` already unlocks a dead owner's mutex through `latch_dead_owner()` and is unchanged.
  > 修复 robust_shm_mutex.hpp try_lock EOWNERDEAD 漏 unlock 及 bpftime_hash_map.hpp 分片单槽位回滚自愈（消除子进程被杀后 pid_events 永久 EBUSY）+ DECISIONS.md
  >
  > 是不是最好的方式, 就是完全不应该毒化? 尽可能不要 boost 容器?
  >
  > HASH/PERCPU_HASH 已经是定长槽位 只把毒化改成"恢复后继续"，替代 #1020 约 10 行 改. 我们是不是默认size 多预留 16 个或者向上取整一下之类的怎样?
  >
  > 你只做 hash 改动, 不需要构建测试, 直接改 pR 就好
  >
  > upsert/erase 两处改用新函数。和 master 相比，生产代码 +50/-10。为啥还是这么多?

- **2026-11-27: generic remote control-plane bridge (`bpftime-remote-host`, `BPFTIME_REMOTE_HOST`), macOS ARM64 preload agent, and zero-fork Linux VM tooling** (#1005, #1006, #1008, #1010, #1013). Unmodified Linux eBPF control-plane tools (`bpftrace`, `bcc`, `libbpf`, `opentelemetry-ebpf-profiler`) trace remote Linux or macOS processes through an OS-neutral client-server control-plane bridge while keeping the data plane (`libbpftime-agent.{so,dylib}`, Frida inline hooks, JIT, maps, ring buffers) in local shared memory on the target host:
  1. **Generic remote protocol (`BPFTIME_REMOTE_HOST`, `BPFTIME_REMOTE_TOKEN`, `BPFTIME_ENABLE_REMOTE_BRIDGE`)**: `libbpftime-syscall-server.so` forwards control operations (`hello`, `proc_check`, `proc_maps`, `fetch_elf`, map/prog/perf_event/link ops, `spawn_cmd`/`resume_cmd`, and bounded `subscribe_events` streaming) over length-prefixed frames to `bpftime-remote-host` (built on both Linux and macOS; Linux serves real `/proc` and target ELF bytes, macOS synthesizes `/proc/<pid>/maps` and 64-bit stub ELFs with `.symtab`/`.note.stapsdt`/ARM64 `.eh_frame`). `bpftime-remote-host` binds `127.0.0.1` by default and requires constant-time 256-bit `BPFTIME_REMOTE_TOKEN` verification on every connection before accepting any operation. When `BPFTIME_REMOTE_HOST` is unset, `libbpftime-syscall-server.so` incurs zero runtime overhead.
  2. **macOS ARM64 preload constructor & optional `BPFTIME_ENABLE_AGENT_IPC_SERVER`**: `libbpftime-agent.dylib` initializes synchronously once from a `dyld` constructor (returning immediately with zero syscalls/files when SHM is absent, zero FIFO/startup waits, reusing the opt-in `BPFTIME_ENABLE_AGENT_IPC` socket at `$TMPDIR/bpftime-agent-<pid>.sock`). On Apple Silicon macOS, `tools/macos_vz_vm/` (`bpftime-vz` + `bpftime-macos`) runs the stock ARM64 Linux control plane inside a rootless `Virtualization.framework` VM on top of `BPFTIME_REMOTE_HOST`.
  3. **Delivery split**: shipped as separate minimal single-purpose pull requests tagged with `otel`, `linux`, and `macos` GitHub labels, alongside updating `example/otel-ebpf-profiler` to run in rootless userspace mode on Linux and macOS.
  > 如何真实的把 bpftrace 和 bcc 以及别的 ebpf 工具, 在 macos 上面用 bpftime 跑起来? 不是替代, 确保真实弄好并且能 work?
  >
  > 控制面跑虚拟机里面是部署更方便?
  >
  > 最好能不用 fork, 但也要真实能追踪 macos 上面的东西?
  >
  > bpftime 本身对 otel ebpf profiler 支持的 example 是不是也要修? 实际上已经基本支持了? 你看看? 是不是也可以测试一下? 在 mac 和 linux 上面?
  >
  > 分开去做; 开的 PR 都要注明是 otel / linux / macos, 的 tag; 尽可能减少不必要的改动
  >
  > 解释一下, 你这个做法本质上是把 bpftime 转化成了 remote 的?
  >
  > 那我们要改名, 确保这个 remote 机制是通用的 remote 机制? 也就是trace 别的机器也行?

- **2026-11-27: remove `has_pending_uprobe` latch and defer `shm_lifecycle_lock` on `dlopen()`** (#970 follow-up; closes #1016). Delete the process-local `has_pending_uprobe` / `has_pending_uprobes()` latch introduced in #970 instead of adding a shared-memory generation counter (#1016): a boolean set only from startup `-ENOENT` misses uprobe links registered in SHM after target startup. On outermost successful non-`RTLD_NOLOAD` host `dlopen()`, the agent takes only non-blocking process-local `try_to_lock(detach_mutex)` + `try_to_lock(ctx_mutex)`, skips already-instantiated and unmapped DSOs (`is_module_mapped`) in memory (`0` syscalls, `0` heap allocations), and lazily acquires shared `shm_lifecycle_lock` (`LOCK_SH | LOCK_NB`) only when an uninstantiated mapped uprobe actually needs to be attached.

- **2026-11-27: the probe-read window is configurable and capped at 512 bytes in static TLS** (#984). `BPFTIME_PROBE_READ_WINDOW_BYTES` (0–512, default 512; 0 disables the window) and `BPFTIME_PROBE_READ_LOOKBEHIND_BYTES` (default 64, at most the window size) are read at startup. The per-thread window stays in `initial-exec` TLS, so a probe read never allocates, and 512 bytes keeps a late `dlopen()` of the agent inside glibc's static-TLS surplus. A larger window needs storage outside TLS and a new decision.
  > bpftime 我们现在 probe read 是不是有缓存? 这个大小长度和方向是不是动态配置更好? 你想想? 这个窗口默认值多少比较好?
  >
  > 这个能配置吗? 我觉得要能动态配置
  >
  > 先限制在 512

- **2026-11-27: no per-CPU execution locks (`bpftime_cpu_exec_locks` / `bpf_cpu_execution_guard`); `PERCPU_ARRAY` stays lock-free with lazy per-invocation CPU ID pinning** (#730, #937). Remove `#730` and `#937`'s global `bpf_cpu_execution_guard` and `bpftime_cpu_exec_locks` from `bpftime_prog_exec` and shared memory (superseding `bpftime_shm::cpu_exec_locks` in L1 below). `bpftime_prog_exec` must never call `sched_getcpu()` or take any per-CPU lock on entry; `PERCPU_ARRAY` stays completely lock-free (`should_lock = false`), matching the pre-`#730` `#298` design and Linux kernel `migrate_disable()` / sleepable-uprobe semantics where per-CPU slots are not mutex-serialized across preempted tasks. Only when a running BPF program actually calls `my_sched_getcpu()` (`PERCPU_*` maps, `bpf_get_smp_processor_id`, `bpf_perf_event_output`) is the logical CPU queried once and cached in a `thread_local int` for the duration of that outermost `bpftime_prog_exec` call (`0` locks, `0` atomics, `0` syscalls, and `0 ns` `sched_getcpu()` cost for programs that do not touch per-CPU state).
  > bpf_cpu_execution_guard 每次无条件调 sched_getcpu() + 跨进程 ROBUST 锁加解锁（第二大头，约 90–130 ns）有必要这个吗? per-CPU 状态的程序 也不需要吧? 我觉得我们用户态不应该考虑 cpu, 而是所有的 cpu 模型都会被映射到thread?
  >
  > sched_getcpu() 在 PERCPU_ARRAY / PERCPU_HASH 里面那是不是也行?
  >
  > 在第一次真正进入 PERCPU_ARRAY / PERCPU_HASH（即 my_sched_getcpu()）时按需调一次并 trylock， 是不是原先就是这样的? 为啥当时改了? 追溯一下完整历史
  >
  > PERCPU_ARRAY 完全不拿锁我觉得要.
  >
  > Scratch 大结构体被抢占踩脏：线程 A 在核 3 调 lookup 拿到 &data[0][3] 写了一半被时钟中断切走，线程 B 被调度上核 3 调 sched_getcpu() 也会拿到同一个 &data[0][3] 把 A 的半截事件覆盖掉，A 恢复后发进 ringbuf 的就是拼装脏数据。内核会出现这样的吗
  >
  > 修了吗? 代码是不是要显著减少? 这个 descision 也需要写进去避免再次弄乱

- **2026-11-27: non-blocking `dlopen()` retry and `L3` lock callsites for unmapped uprobes** (#970). When a Frida uprobe target DSO is not mapped at startup (`-ENOENT`), `bpf_attach_ctx` sets an atomic `has_pending_uprobes` flag and retries pending uprobe links on outermost successful non-`RTLD_NOLOAD` host `dlopen()` calls in the initialized PID, clearing `has_pending_uprobes` once no unmapped uprobe links remain so steady-state `dlopen()` returns before taking any lock or opening any lock file. While `has_pending_uprobes` is set, the `dlopen()` path uses three non-blocking `L3` lock callsites: (1) `detach_mutex` (`std::try_to_lock`) to avoid racing session teardown; (2) shared `shm_lifecycle_lock` (`LOCK_SH | LOCK_NB`) so `dlopen()` never scans SHM across concurrent `syscall-server` segment recreation or final unlink; and (3) `ctx_mutex` (`std::try_to_lock`) to serialize `bpf_attach_ctx` state without blocking the host loader.
  > 测试过了 pytorch 的 cache allocatpr 能 work? 去测一下, 可以的话就这样 pr
  >
  > 最好也要 alphasedd 测试一下
  >
  > alpohaseed merlin
  >
  > 帮忙修掉

- **2026-11-26: compile-time shared-memory type gate and split series for per-process handle tables** (#959, #987). Objects placed in `boost::interprocess::managed_shared_memory` go through `bpftime::shm_construct` / `bpftime::shm_find_or_construct` (`runtime/include/bpftime_shm_safe.hpp`), where `shm_safe_type<T>` enforces at compile time: no `virtual` functions/destructors or `vptr`, no raw/member pointers (`mmap` addresses, `CUcontext`/`CUstream` handles, host strings), no C++ references, and no process-local STL heap containers or `std::mutex`; SHM constructors take a non-copyable `shm_init_ctx` instead of storing `managed_shared_memory &`. Plain `int` OS file descriptors (`map_fd`, `pfd`) are forbidden in SHM by review rule. Per single-responsibility and anti-over-engineering rules, the work lands as a 3-PR series without speculative metaprogramming or open-addressed hash tables: (a) `#987` for the minimal compile-time gate, `shm_init_ctx`, removal of the 7 `virtual` map destructors, and `uintptr_t` CUDA response layout; (b) CPU kernel-user maps moved to a load-time-`pthread_atfork`, O(1) direct-indexed `.bss` slot table; (c) GPU/CUDA maps and `bpf_attach_ctx_cuda` separated for GPU CI.
  > 创建进程局部引用 这个事情 在 bpftime 里面要不要规定一下? 是不是很多问题来源于?
  >
  > 有没有办法用类型阻止这个事情发生? 说说完整的设计, 去把 959 扩展一下, 变成一个完整的修复, 并且检查所有的 PR 里面有没有
  >
  > 这个会违规引入 syscall 或者说之类的吗? ring buffer 我们会把 906 完全 revert
  >
  > 扩展 #959 的实现范围更收敛：新增单头文件 runtime/include/bpftime_shm_safe.hpp（shm_safe_type<T>、shm_construct、shm_init_ctx、零 syscall 的 .bss process_local_cache），同步更新 DECISIONS.md 与 AGENTS.md；去掉 7 个 SHM map 类的 virtual 析构；将 array_map_kernel_user、array_map_kernel_gpu、hash_map_kernel_user、percpu_array_map_kernel_user、perf_event_array_kernel_user、nv_gpu_shared_array_map 中的 map_fd / pfd / mmap_ptr / CUcontext / CUstream 全部迁出 SHM 到 .bss 局部表（完全不碰 ringbuf_map.*）；对外部 PR #328（SHM args.uprobe_multi 残留用户态指针）、#661（QNX tid 缺 atfork 重置）、#703（shm_lifecycle_lock 含 sleep_for 轮询）仅提交英文技术 Review 意见。去做
  >
  > bpftime SHM 编译期类型安全 PR（已创建并 CI 全绿） 过大了? 过度设计? 你也看看 commet
  >
  > 去修, 你来修

- **2026-11-26: single-file SHM lock gateway (`runtime/include/robust_shm_mutex.hpp`), no global lock on per-CPU or primary hash/array/bloom maps, explicit L0–L3 lock inventory, and mandatory approval for any lock change** (#958, #988, #990).
  1. **Single lock file (`runtime/include/robust_shm_mutex.hpp`):** all cross-process shared-memory lock primitives, RAII guards, blocking-vs-`trylock` dispatch (`with_robust_shm_lock`), and skip counters live exclusively in `runtime/include/robust_shm_mutex.hpp`. Defining or calling raw cross-process locks (`pthread_mutex_*`, `pthread_spin_*`, `boost::interprocess::interprocess_*mutex`, custom spinlocks) or bespoke lock-policy wrappers outside `runtime/include/robust_shm_mutex.hpp` is forbidden.
  2. **No global `map_lock` on per-CPU maps, fixed-slot/atomic maps, or `HASH` / `PERCPU_HASH`:** `PERCPU_ARRAY`, `PERCPU_HASH`, `ARRAY`, `PROG_ARRAY`, `PERF_EVENT_ARRAY`, `BLOOM_FILTER`, `RINGBUF`, and `HASH` must all keep `should_lock = false`. `PERCPU_HASH` shares `bpftime_hash_map`'s preallocated fixed-slot array (with Boost `unordered_map` and `key_templates` removed) so `lookup` and per-CPU in-place updates are lock-free and Four-Zeros-compliant.
  3. **Approved L0–L3 lock layering inventory:**
     - **L0 (lock-free atomics / fixed arrays / seqlock):** `epoch_state` seqlock, `atomic_pid_set`, `ARRAY`, `PERCPU_ARRAY`, `PROG_ARRAY`, `PERF_EVENT_ARRAY`, `BLOOM_FILTER`, `RINGBUF` consumer & submit paths, `HASH` & `PERCPU_HASH` `lookup` / iteration / per-CPU slice updates, and `software_perf_event_data` steady-state TLS shard cache;
     - **L1 (short pointer-free SHM mutual exclusion via `robust_shm_mutex::lock_and_recover`):** `bpftime_shm::cpu_exec_locks` and `ringbuf::reserve_lock`;
     - **L2 (poison-on-death `robust_shm_mutex` via `with_robust_shm_lock` — `try_lock` on `LD_PRELOAD` helper hot paths, blocking `lock` on control-plane/consumer paths):** `bpftime_hash_map` 64 sharded write stripes (`HASH` / `PERCPU_HASH` `update` & `delete`), `bpf_map_handler::map_lock` (`LRU_HASH`, `LPM_TRIE`, `QUEUE`, `STACK`, `STACK_TRACE`, `HASH_OF_MAPS`), and cold-start `software_perf_event_data::shard_lock`;
     - **L3 (control-plane / process-local):** `shm_lifecycle_lock` (`flock` around `/dev/shm` create/destroy) and process-local `std::mutex` (`ctx_mutex`, `g_cuda_watcher_mutex`, GDRCopy mapping state).
  4. **Explicit reason & approval gate:** adding any new lock, adding a new lock callsite outside the inventory above, or changing an existing lock's layer, granularity, or recovery policy requires an explicit reason recorded in `DECISIONS.md` and prior maintainer approval before merge.
  > 我们要确保仓库里面加任何锁都需要有明确的原因记录和批准流程, 不然一直出错?
  >
  > 我们是不是锁需要有明确的分层, 半年前的基础上增量改进?
  >
  > 整体代码需要增加多少? 100 行以内, 包含测试搞定? 都放在一个文件里面, 禁止在这个文件之外加锁?
  >
  > 按你说的去都改掉. 批准把 PERCPU_HASH 统一收敛到 bpftime_hash_map 的预分配定长槽位数组上（净删掉 per_cpu_hash_map 里的 Boost 哈希表与 key_template. 就在这个 PR 里面改; 文档的也要改, per cpu 这种都不要全局锁


- **2026-11-26: forbid speculative lifecycle/GC state machines, test-only control-plane/IPC backdoors, and parallel rewrites of idempotent paths; close `#971` into minimal `#970`** (#970, #971). Three forms of speculative bloat are banned across the codebase and treated as `over-engineering` in review:
  1. **No unrequested symmetric teardown/GC or hypothetical edge-case state machines** without a real production caller or reproduced failure (e.g. `#971`'s runtime SHM link-deletion teardown pass on host `dlopen()` and its heap-allocating `live_uprobe_targets` `std::set` for duplicate same-offset links);
  2. **No control-plane/IPC commands, flags, or hooks added solely to drive tests** when unused in default production modes (e.g. `#971`'s UNIX socket `"reconcile"` command while `BPFTIME_ENABLE_AGENT_IPC` defaults to off under bare `LD_PRELOAD`);
  3. **No parallel `reconcile_*` / `refresh_*` rewrites when an existing path is already idempotent** (e.g. `bpf_attach_ctx::instantiate_handler_at()` already skips `instantiated_handlers.contains(id)` in `O(1)`, so `dlopen()` retry wraps it thinly instead of duplicating a 130-line state machine).
  `#971` is closed, and only the minimal non-blocking `dlopen()` retry of unattached `-ENOENT` uprobes is folded into `#970` alongside skipping `gum_module_load()` on Linux.
  > #971 里手搓的 30 行运行期反向卸载 GC、70 行 live_uprobe_targets std::set（每次有待挂项都做堆分配）、以及默认根本不开的 UNIX Socket IPC "reconcile" 命令协议，全都可以直接删掉。如何避免这些再发生? 我们是不是要有 rule / desicision 阻止这个. 去合并并且压缩, 关掉 971
- **2026-11-26: the other hot-path items of the entry from #977 are approved** (#977, #979). These four, listed there without the maintainer's words, now stand:
  1. `process_vm_writev` on every `bpf_probe_write_user` in the default build (`ENABLE_PROBE_WRITE_CHECK` defaults to `ON`);
  2. one-time identity caching: `SYS_gettid` once per thread into TLS and `readlink("/proc/self/exe")` once per process, results cached, host `errno` preserved;
  3. vDSO `clock_gettime` (`bpf_ktime_get_ns` and the other time helpers), including its fallback to a real syscall on clock sources the vDSO does not serve;
  4. no `static` heap-owning C++ containers in injected code, since their `__cxa_atexit` destructors run at `exit()` while other threads may still use them.
  > 无原话支持的豁免 是什么? 我应该支持, 你觉得合理吗
  >
  > 明确记住, 全部同意
- **2026-11-26: `process_vm_readv` for probe reads in the default build is allowed** (#977, #979). `ENABLE_PROBE_READ_CHECK` defaults to `ON` (`cmake/StandardSettings.cmake`), so in a default build `bpf_probe_read*` reaches `process_vm_readv`: inside an attach event once per miss of the per-thread 128-byte read-ahead window (`PROBE_READ_AHEAD_CAP` in `runtime/src/bpf_helper.cpp`), outside an event once per read. That call is an approved hot-path syscall; it replaces the "opt-in ... explicitly enabled" wording of item 2 in the entry below for `process_vm_readv`.
  > 默认构建每次 probe read 都会调用它 没问题
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
