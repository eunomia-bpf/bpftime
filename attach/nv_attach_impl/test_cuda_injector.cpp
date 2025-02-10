#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>
#include <cstdint>

// Include your CUDAInjector header:
#include "nv_attach_impl.hpp"
TEST_CASE("Test CUDAInjector - basic attach/detach")
{
    // For demonstration, pick a dummy or real PID.
    // In a real-world test, you'd spawn a child process running a CUDA app.
    pid_t test_pid = 12345; 

    // 1. Construct the injector
    bpftime::attach::CUDAInjector injector(test_pid);

    // 2. Attempt to attach to the process
    bool attached = injector.attach();
    REQUIRE(attached == true);

    // 3. [Optional] Attempt to inject PTX code
    SECTION("Inject PTX code")
    {
        // A trivial PTX kernel as an example
        const char* ptx_code = R"PTX(
        .version 7.0
        .target sm_70
        .address_size 64

        .visible .entry injected_kernel() {
            // A do-nothing kernel
            ret;
        }
        )PTX";

        // Suppose we want to inject at some device memory address (dummy).
        CUdeviceptr dummy_inject_addr = 0x10000000;  
        size_t dummy_code_size        = 256; // Example

        // A hypothetical method in CUDAInjector for demonstration
        bool success = injector.inject_ptx(ptx_code, dummy_inject_addr, dummy_code_size);
        REQUIRE(success == true);
    }

    // 4. Detach from the process
    bool detached = injector.detach();
    REQUIRE(detached == true);

    // 5. Attempting to attach again or inject code after detaching might fail
    //    but you can add negative tests if you want:
    SECTION("Attach again (negative test)")
    {
        bool reattach = injector.attach();
        REQUIRE(reattach == false); 
    }
}
