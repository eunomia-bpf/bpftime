# bpftime-verifier

A simple wrapper around ebpf-verifier, providing more simple user interfaces.

## Usage

### Setup cmake
```
add_subdirectory(bpftime-verifier)

add_dependencies(MY_TARGET bpftime-verifier)
target_link_libraries(MY_TARGET bpftime-verifier)
target_include_directories(MY_TARGET PUBLIC ${BPFTIME_VERIFIER_INCLUDES}s)
```
### Invokes the verifier

```cpp
#include <bpftime-verifier.hpp>
#include <map>
#include <iostream>
#include <optional>
#include <string>
using namespace bpftime;

int main(){
    // Set maps that the current ebpf program will use
    set_map_descriptors(std::map<int, BpftimeMapDescriptor>{
        { 2333, BpftimeMapDescriptor{ .original_fd = 233,
                            .type = BPF_MAP_TYPE_HASH,
                            .key_size = 8,
                            .value_size = 4,
                            .max_entries = 8192,
                            .inner_map_fd = 0 } } });

    // Set helpers that the current ebpf program will use
    // This can include both kernel-provided helpers and self-defined helpers
    set_available_helpers(std::vector<int32_t>{ 1, 1000001 });
    // For user-defined helpers, prototype should also be defined
    set_non_kernel_helpers(std::map<int, BpftimeHelperProrotype>{
        {1000001, BpftimeHelperProrotype{
            .name = "my_helper",
            .return_type = EBPF_RETURN_TYPE_INTEGER,
            .argument_type = {
                EBPF_ARGUMENT_TYPE_ANYTHING, // This indicates an arbiraty 64bit integer
                EBPF_ARGUMENT_TYPE_PTR_TO_MAP, // This indicates a pointer of map
            }
        }}
    });

    const uint64_t prog_with_map_1[] = { 
        0x00000002000001b7, 
        0x00000000fff81a7b,
        0x000000000000a2bf, 
        0xfffffff800000207,
        0x0000091d00001118, 
        0x0000000000000000,
        0x0000000100000085, 
        0x0000000000000061,
        0x0000000000000095 };

    // Do the verification
    std::optional<std::string> result = auto ret =
            verify_ebpf_program(prog_with_map_1, std::size(prog_with_map_1),
                        "uprobe//proc/self/exe:uprobed_sub");

    // If verification succeeded, verify_ebpf_program will return an empty optional. Otherwise, failure message will be returned
    if(result.has_value()){
        std::cerr << result.value();
    } else {
        std::cout << "Done!";
    }
}
```

## Important notes
Things set by `set_available_helpers`, `set_non_kernel_helpers` and `set_map_descriptors` are thread-local, meaning that each thread has its own instance of things set by these functions. So you need to set the corresponding values in each thread you use.
