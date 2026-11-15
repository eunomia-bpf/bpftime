// SPDX-License-Identifier: MIT
// RAII write windows for the pkey-protected shared-memory segment.
//
// When BPFTIME_ENABLE_MPK is built in and a protection key was allocated,
// an SHM segment is mapped write-disabled by default. Any code that mutates
// state inside it must run inside a write-enabled window or it takes a
// SIGSEGV on the first store.
//
// Two windows share one per-thread nesting counter so nested writers never
// re-protect a segment while an outer window is still active:
//   * mpk_global_write_guard toggles the process-global singleton segment
//     (used around BPF program execution, whose helpers write that segment);
//   * bpftime_shm::mpk_instance_write_guard toggles the specific shm
//     instance's own key, which is required because protection keys are
//     per-bpftime_shm instance and the global toggle cannot reach a
//     separately constructed segment.
//
// When MPK is compiled out, or pkey allocation failed at runtime
// (is_mpk_init == false), the protect calls are no-ops, leaving only a
// thread-local increment/decrement.
#pragma once

#include "bpftime_shm.hpp"

#if BPFTIME_ENABLE_MPK
namespace bpftime
{
// Single depth shared by every guard type in the thread.
inline thread_local int mpk_write_window_depth = 0;

template <class DisableFn, class EnableFn>
class scoped_mpk_write_window
{
      public:
	scoped_mpk_write_window(DisableFn disable, EnableFn enable)
		: enable_fn(enable)
	{
		if (mpk_write_window_depth++ == 0)
			disable();
	}
	~scoped_mpk_write_window()
	{
		if (--mpk_write_window_depth == 0)
			enable_fn();
	}
	scoped_mpk_write_window(const scoped_mpk_write_window &) = delete;
	scoped_mpk_write_window &
	operator=(const scoped_mpk_write_window &) = delete;

      private:
	EnableFn enable_fn;
};

// Window around the process-global singleton segment.
class mpk_global_write_guard
	: public scoped_mpk_write_window<void (*)(), void (*)()>
{
      public:
	mpk_global_write_guard()
		: scoped_mpk_write_window<void (*)(), void (*)()>(
			  bpftime_protect_disable, bpftime_protect_enable)
	{
	}
};
} // namespace bpftime

#define BPFTIME_MPK_GLOBAL_WRITE_GUARD() \
	::bpftime::mpk_global_write_guard bpftime_mpk_global_guard
#else
#define BPFTIME_MPK_GLOBAL_WRITE_GUARD() ((void)0)
#endif
