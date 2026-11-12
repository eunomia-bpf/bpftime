// Shared helpers for the trap backend tests
#ifndef _BPFTIME_TRAP_TEST_COMMON_HPP
#define _BPFTIME_TRAP_TEST_COMMON_HPP
// Functions that receive probes must stay a single, out-of-line copy: with
// -O2 GCC otherwise emits constprop/isra clones and calls those instead of
// the symbol the probe was placed on.
#if defined(__GNUC__) && !defined(__clang__)
#define TRAP_TEST_TARGET __attribute__((noinline, noclone, used))
#else
#define TRAP_TEST_TARGET __attribute__((noinline, used))
#endif
#endif
