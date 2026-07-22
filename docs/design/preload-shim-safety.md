# LD_PRELOAD shim safety and transparency

bpftime loads the syscall server, agent, and text-segment transformer into a
process that it does not own. These components share the host's address space,
file descriptors, environment, standard streams, signal state, and lifetime.
Their error policy is therefore stricter than the policy for a standalone CLI.

This document applies to:

- `libbpftime-syscall-server.so` from `runtime/syscall-server/`;
- `libbpftime-agent.so` from `runtime/agent/`;
- `libbpftime-agent-transformer.so` from
  `attach/text_segment_transformer/`; and
- runtime helpers that execute inside a process containing one of these
  libraries.

## Host-preservation invariants

An internal bpftime error must not call `exit`, `_exit`, `abort`, `terminate`,
or otherwise end the host process. It must not let a C++ exception cross an
exported C function or interposition boundary. Recoverable initialization and
interception failures must fail open: disable the affected bpftime behavior and
delegate to the original host function or syscall with the original arguments.

Wrappers must resolve and retain original functions without using assertions as
error handling. If a wrapper cannot provide bpftime behavior, it should preserve
the host operation's return value and `errno` behavior by invoking the original
operation. A bpftime-specific error is appropriate only for a bpftime-specific
entry point that has no host operation to delegate to.

Partial text transformation is a special case. Until the agent callback is
installed, rewritten syscall sites must continue through the original syscall
instruction. A later setup failure may leave the transparent trampoline in
place, but it must not redirect calls into partially initialized bpftime state.

These requirements do not hide or alter a termination, signal, exception, or
standard-stream write performed by the host application itself.

## Logging and standard streams

Injected code must not write directly to stdout or stderr. This includes error
paths, destructors, tracing helpers, third-party logging fallbacks, and early
initialization before the configured logger is ready.

The default log destination is the configured rotating log file. Console
logging is opt-in and is enabled only when `BPFTIME_LOG_OUTPUT=console`. If the
configured sink cannot be created, logging becomes silent; it must not fall back
to a console sink and must not terminate the host. Formatting or flushing a log
message is also best effort and must not allow an exception to escape.

Program output requested through helpers such as `bpf_trace_printk` is routed
through the bpftime logger. It reaches stderr only when console logging was
explicitly selected.

## Implementation checklist

For every new or changed preload wrapper:

1. Put a non-throwing boundary around bpftime-owned work.
2. Keep the original function or syscall available as the fallback.
3. Disable further interception when bpftime state is no longer trustworthy.
4. Preserve variadic calling conventions. Where API flags expose whether an
   optional argument exists, as with `open` and `openat`, do not read it unless
   it is present.
5. Avoid direct standard I/O, assertions, and process-termination calls.
6. Treat logger creation, logging, and teardown as fallible operations.

Regression tests should execute a small host program with each affected
library in `LD_PRELOAD`, force the relevant failure, and verify both the host's
exit status and captured stdout/stderr. At minimum, coverage should include
shared-memory exhaustion, an unavailable log sink, missing agent state, and
transformer setup failure.
