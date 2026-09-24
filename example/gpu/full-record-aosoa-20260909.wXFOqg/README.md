# Retained full-record experiment source

This directory retains the exact local build inputs collected on 2026-09-24.
Its dated directory name is preserved because historical commands refer to it.
The C sources and Makefile have not been changed during collection.
Generated `.output*` directories and executables stay local and are ignored.

Build from this directory, selecting the corresponding `LAYOUT` shown below.
The Makefile expects bpftime dependencies at `../../../third_party`, CUDA
headers/libraries and a BPF-capable Clang. The default layout is `aos`.

```sh
make LAYOUT=aosoa
```

Collection verifies the source inventory and Git patch formatting only; it
adds no new GPU measurements or build claims. Existing measurements and build
records are in the sibling gpu_ext repository under
`workloads/llama.cpp/observability_overhead/revision-rq4/`:

- `results-full-record-soa-20260908.md`
- `full-record-device-buffer/aosoa-regression-codegen-analysis-20260909.md`
- `results-full-record-soa-warp-20260909.md`
