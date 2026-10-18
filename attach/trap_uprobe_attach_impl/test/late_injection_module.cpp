#include <trap_uprobe_attach_impl.hpp>
#include <atomic>
#include <cstdlib>
#include <pthread.h>

using namespace bpftime;
using namespace bpftime::attach;
using namespace bpftime::attach::trap;

namespace {
std::atomic<unsigned> unsafe_calls{ 0 };
trap_attach_impl *manager = nullptr;
int entry_id = -1, return_id = -1;
void audit()
{
	if (signal_handler_depth)
		unsafe_calls.fetch_add(1, std::memory_order_relaxed);
}
}

// Link-time interception is confined to this test DSO and its static
// dependencies. Calls in the host's allocator and test framework are excluded.
extern "C" void *__real_malloc(size_t);
extern "C" void *__wrap_malloc(size_t n) { audit(); return __real_malloc(n); }
extern "C" void *__real_calloc(size_t, size_t);
extern "C" void *__wrap_calloc(size_t n, size_t s) { audit(); return __real_calloc(n, s); }
extern "C" void *__real_realloc(void *, size_t);
extern "C" void *__wrap_realloc(void *p, size_t n) { audit(); return __real_realloc(p, n); }
extern "C" void __real_free(void *);
extern "C" void __wrap_free(void *p) { audit(); __real_free(p); }
extern "C" void *__real__Znwm(size_t);
extern "C" void *__wrap__Znwm(size_t n) { audit(); return __real__Znwm(n); }
extern "C" int __real_pthread_mutex_lock(pthread_mutex_t *);
extern "C" int __wrap_pthread_mutex_lock(pthread_mutex_t *p)
{
	audit();
	return __real_pthread_mutex_lock(p);
}
extern "C" void *__real___tls_get_addr(void *);
extern "C" void *__wrap___tls_get_addr(void *p)
{
	audit();
	return __real___tls_get_addr(p);
}

extern "C" int install_probes(void *target, std::atomic<unsigned> *hits)
{
	// Positive control in ordinary context: prove the malloc wrapper really
	// counts before relying on a zero result from the actual signal handlers.
	++signal_handler_depth;
	void *sample = malloc(32);
	asm volatile("" : : "r"(sample) : "memory");
	--signal_handler_depth;
	free(sample);
	if (unsafe_calls.exchange(0) == 0)
		return -EIO;
	spdlog::set_level(spdlog::level::trace);
	manager = new trap_attach_impl;
	entry_id = manager->create_uprobe_at(target, [hits](const pt_regs &) {
		uint64_t value = 0, worker = 4;
		if (bpftime_trap_get_func_arg(0, 1, &worker, 0, 0) == 0 && worker < 4)
			hits[worker + 1].fetch_add(1, std::memory_order_relaxed);
		// Exercise helper failure paths at Debug log level too.
		if (bpftime_trap_get_func_arg(0, 99, &value, 0, 0) == (uint64_t)-EINVAL &&
		    bpftime_set_retval(1) == (uint64_t)-EINVAL)
			hits[0].fetch_add(1, std::memory_order_relaxed);
	});
	if (entry_id < 0)
		return entry_id;
	return_id = manager->create_uretprobe_at(target, [hits](const pt_regs &) {
		uint64_t value = 0;
		if (bpftime_trap_get_func_ret(0, &value, 0, 0, 0) == 0 && value == 42)
			hits[0].fetch_add(1, std::memory_order_relaxed);
	});
	return return_id < 0 ? return_id : 0;
}

extern "C" int remove_probes()
{
	int entry = manager->detach_by_id(entry_id);
	int ret = manager->detach_by_id(return_id);
	delete manager;
	manager = nullptr;
	return entry < 0 ? entry : ret;
}

extern "C" unsigned unsafe_operation_count() { return unsafe_calls.load(); }
