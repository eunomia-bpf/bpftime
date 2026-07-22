# Repository maintenance guidance

Follow `CONTRIBUTING.md` and the component-specific documentation before
changing this repository. Keep changes scoped, add regression coverage for
behavior changes, and run the smallest relevant build and test targets before
publishing a pull request.

## Injected and preloaded code

Changes to `runtime/syscall-server/`, `runtime/agent/`, or
`attach/text_segment_transformer/` must preserve the host process. Read and
follow [`docs/design/preload-shim-safety.md`](docs/design/preload-shim-safety.md)
before editing those components.

In particular, an internal bpftime failure must not terminate the host, escape
through a C ABI boundary as a C++ exception, replace a recoverable host
operation with a bpftime-only failure, or write to the host's stdout or stderr
unless the user explicitly selected console logging.
