#!/usr/bin/env python3
"""CPU-only regression gates for late CUDA attach lifetime invariants."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[3]
IMPL = (ROOT / "attach/nv_attach_impl/nv_attach_impl.cpp").read_text()
HEADER = (ROOT / "attach/nv_attach_impl/nv_attach_impl.hpp").read_text()
FRIDA = (ROOT / "attach/nv_attach_impl/nv_attach_impl_frida_setup.cpp").read_text()
FATBIN = (ROOT / "attach/nv_attach_impl/nv_attach_fatbin_record.cpp").read_text()


def section(source: str, start: str, end=None) -> str:
    begin = source.index(start)
    finish = len(source) if end is None else source.index(end, begin + len(start))
    return source[begin:finish]


def compact(source: str) -> str:
    return re.sub(r"\s+", "", source)


class LateAttachSourceInvariants(unittest.TestCase):
    def test_device_verifier_guard_and_timer_boundaries(self):
        body = section(
            IMPL,
            "int nv_attach_impl::create_attach_with_ebpf_callback(",
            'extern "C" {',
        )
        unavailable = body.index("GPU eBPF verifier unavailable")
        for mutation in (
            "for (const auto &pd : this->pass_configurations)",
            "enabled.store(true",
            "this->allocate_id()",
            "start_late_bootstrap_async()",
        ):
            self.assertLess(unavailable, body.index(mutation))
        self.assertIn("policy_entry_created=0", body[: body.index("#endif", unavailable)])

        maps = body.index("const auto map_descriptors")
        started = body.index("const auto verification_started")
        verify = body.index("verify_gpu_program(")
        stopped = body.index("measured_verification_elapsed_ns")
        timing = body.index("GPU eBPF verification timing:")
        outcome = body.index("if (error)")
        self.assertLess(maps, started)
        self.assertLess(started, verify)
        self.assertLess(verify, stopped)
        self.assertLess(stopped, timing)
        self.assertLess(timing, outcome)
        self.assertEqual(body.count("verify_gpu_program("), 1)
        self.assertEqual(body.count("verification_elapsed_ns={}"), 1)
        self.assertIn("std::chrono::steady_clock::now()", body[started:verify])
        self.assertIn("std::chrono::steady_clock::now()", body[verify:stopped + 300])
        self.assertIn("std::max<int64_t>", body[stopped:timing])

        # Keep frozen admission messages stable; timing is a separate line.
        self.assertIn(
            "(mode=STRICT, policy_entry_created=0)", body[outcome:]
        )
        self.assertIn("verification failed for {}: {}; continuing", body[outcome:])
        self.assertIn(
            "verification accepted: mode=STRICT program={} attach={} instructions={}",
            body[outcome:],
        )

    def test_late_bootstrap_registers_only_requested_hook_targets(self):
        body = section(
            IMPL,
            "void nv_attach_impl::prefill_patched_kernel_functions_from_loaded_fatbins()",
            "\nnamespace\n{",
        )
        self.assertIn("const auto kernels = collect_all_kernels_to_patch();", body)
        self.assertIn("for (const auto &kernel : kernels)", body)
        self.assertIn("record_patched_kernel_function(kernel", body)
        self.assertNotIn("original_ptx", body)
        self.assertNotIn("collect_ptx_entry_functions", IMPL)

    def test_driver_launch_records_completion_after_original_launch(self):
        body = section(
            FRIDA,
            "static CUresult cu_launch_kernel_common(",
            'extern "C" CUresult cuda_driver_function__cuLaunchKernel(',
        )
        patched = body.index("func = *patched;")
        suffix = body[patched:]
        original = suffix.index("original(")
        event = suffix.index("record_patched_launch(")
        returned = suffix.index("return", event)
        self.assertLess(original, event)
        self.assertIn("CUDA_SUCCESS", suffix[original:event])
        self.assertLess(event, returned)

    def test_graph_wrappers_forward_all_non_function_arguments(self):
        wrappers = (
            (
                "cuda_driver_function__cuGraphAddKernelNode_v1(",
                "cuda_driver_function__cuGraphAddKernelNode_v2(",
                "original(phGraphNode,hGraph,dependencies,numDependencies,nodeParams);",
            ),
            (
                "cuda_driver_function__cuGraphAddKernelNode_v2(",
                "cuda_driver_function__cuGraphExecKernelNodeSetParams_v1(",
                "original(phGraphNode,hGraph,dependencies,numDependencies,nodeParams);",
            ),
            (
                "cuda_driver_function__cuGraphExecKernelNodeSetParams_v1(",
                "cuda_driver_function__cuGraphExecKernelNodeSetParams_v2(",
                "original(hGraphExec,hNode,nodeParams);",
            ),
            (
                "cuda_driver_function__cuGraphExecKernelNodeSetParams_v2(",
                "cuda_driver_function__cuGraphKernelNodeSetParams_v1(",
                "original(hGraphExec,hNode,nodeParams);",
            ),
            (
                "cuda_driver_function__cuGraphKernelNodeSetParams_v1(",
                "cuda_driver_function__cuGraphKernelNodeSetParams_v2(",
                "original(hNode,nodeParams);",
            ),
            (
                "cuda_driver_function__cuGraphKernelNodeSetParams_v2(",
                "static cudaError_t mirror_cuda_memcpy_from_symbol(",
                "original(hNode,nodeParams);",
            ),
        )
        for start, end, expected in wrappers:
            with self.subTest(wrapper=start):
                body = section(FRIDA, start, end)
                self.assertIn(expected, compact(body))
                self.assertNotIn("params_to_use", body)
                self.assertNotIn("active_impl", body)

    def test_callbacks_borrow_and_teardown_revokes_active_implementation(self):
        context = section(
            HEADER,
            "struct CUDARuntimeFunctionHookerContext {",
            "struct nv_attach_entry {",
        )
        self.assertIn("uint64_t generation", context)
        self.assertNotIn("nv_attach_impl *impl", context)
        self.assertNotIn("active_impl.load", FRIDA)

        callback_markers = (
            ("static void example_listener_on_enter(", "static void example_listener_on_leave("),
            ("static void example_listener_on_leave(", "static void\ncuda_runtime_function_hooker_class_init("),
            ('extern "C" cudaError_t\ncuda_runtime_function__cudaLaunchKernel(', "static std::optional<std::string>\ncuda_graph_maybe_get_kernel_name_from_cufunction("),
            ('extern "C" CUresult cuda_driver_function__cuLaunchKernel(', 'extern "C" cudaError_t cuda_runtime_function__cudaLaunchKernel_ptsz('),
            ('extern "C" cudaError_t cuda_runtime_function__cudaLaunchKernel_ptsz(', "static const CUDA_KERNEL_NODE_PARAMS_v1 *"),
            ('extern "C" cudaError_t cuda_runtime_function__cudaMemcpyFromSymbol(', 'extern "C" cudaError_t cuda_runtime_function__cudaMemcpyFromSymbolAsync('),
            ('extern "C" cudaError_t cuda_runtime_function__cudaMemcpyFromSymbolAsync(', None),
        )
        for start, end in callback_markers:
            with self.subTest(callback=start):
                body = section(FRIDA, start, end)
                lock = body.index("std::shared_lock<std::shared_mutex>")
                borrow = body.index("state.active_impl") if "state.active_impl" in body else body.index("state->active_impl")
                self.assertLess(lock, borrow)

        listener = section(
            FRIDA,
            "static void example_listener_on_enter(",
            "static void example_listener_on_leave(",
        )
        self.assertIn("context->generation != state.active_generation", listener)

        constructor = section(
            IMPL, "nv_attach_impl::nv_attach_impl()", "nv_attach_impl::~nv_attach_impl()"
        )
        self.assertIn("std::unique_lock<std::shared_mutex>", constructor)
        self.assertIn("hook_generation = ++hook_state.next_generation", constructor)
        self.assertIn("hook_state.listener_contexts.push_back", constructor)
        publish = constructor.rindex("hook_state.active_impl = this")
        self.assertLess(constructor.index("this->ptx_compiler = *compiler"), publish)
        self.assertLess(
            publish, constructor.index("hook_state.replacements_installed.store", publish)
        )
        destructor = section(IMPL, "nv_attach_impl::~nv_attach_impl()", "void nv_attach_impl::record_patched_kernel_function(")
        revoke = destructor.index("hook_state.active_impl = nullptr")
        self.assertIn("std::unique_lock<std::shared_mutex>", destructor)
        self.assertLess(revoke, destructor.index("hook_state.active_generation = 0"))
        self.assertLess(
            revoke,
            destructor.index("gum_interceptor_detach"),
        )

    def test_worker_failures_are_aggregated_after_join(self):
        workers = (
            section(FATBIN, "fatbin_record::compile_ptxs(", "void fatbin_record::try_loading_ptxs("),
            section(IMPL, "nv_attach_impl::hack_fatbin(", "int nv_attach_impl::find_attach_entry_by_program_name("),
        )
        for body in workers:
            with self.subTest(worker=body.splitlines()[0]):
                self.assertIn("std::exception_ptr", body)
                self.assertIn("std::current_exception()", body)
                self.assertIn("std::rethrow_exception(", body)
                self.assertLess(body.index("pool.join()"), body.index("std::rethrow_exception("))

    def test_detach_timeout_preserves_live_state(self):
        detach = section(IMPL, "int nv_attach_impl::detach_by_id(", "void nv_attach_impl::register_custom_helpers(")
        wait = section(IMPL, "bool nv_attach_impl::wait_for_patched_launch_events(", "void nv_attach_impl::clear_patched_state_for_next_session()")
        self.assertLess(detach.index("!wait_for_patched_launch_events"), detach.index("hook_entries.erase(itr)"))
        self.assertLess(detach.index("return -EBUSY"), detach.index("hook_entries.erase(itr)"))
        self.assertLess(detach.index("hook_entries.erase(itr)"), detach.index("clear_patched_state_for_next_session()"))
        self.assertNotIn("pending_launch_events_by_stream.clear()", wait)
        self.assertIn("return false;", wait)

    def test_fatbin_records_follow_runtime_handles(self):
        enter = section(FRIDA, "static void example_listener_on_enter(", "static void example_listener_on_leave(")
        leave = section(FRIDA, "static void example_listener_on_leave(", "static void\ncuda_runtime_function_hooker_class_init(")
        self.assertIn("registering_fatbin = fatbin_record.get()", enter)
        self.assertGreaterEqual(enter.count("fatbin_handle_to_record.find(fatbin_handle)"), 2)
        self.assertIn("fatbin_handle_to_record.erase(fatbin_handle)", enter)
        self.assertIn("gum_invocation_context_get_return_value", leave)
        self.assertIn("fatbin_handle_to_record[handle]", leave)
        self.assertIn("registering_fatbin = nullptr", leave)


if __name__ == "__main__":
    unittest.main(verbosity=2)
