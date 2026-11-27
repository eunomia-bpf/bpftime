/* SPDX-License-Identifier: MIT
 *
 * Darwin (Apple Silicon) support for the injected bpftime agent.
 *
 * Every entry point here is single-shot, synchronous and bounded. There is
 * no background polling and no FIFO/blocking handshake: the spawner suspends
 * the child before dyld runs and only resumes it once the shared-memory
 * segment and handlers are ready, so the agent probes shared memory exactly
 * once and stays inert when it is absent.
 */
#ifndef BPFTIME_DARWIN_AGENT_HPP
#define BPFTIME_DARWIN_AGENT_HPP

#include <string>

namespace bpftime
{
namespace agent_darwin
{

// Filesystem location of the dyld image table exported to the host daemon,
// e.g. /tmp/bpftime-maps-12345.txt.
std::string maps_file_path(int pid);

// Atomically (tmp file + rename(2)) write the currently loaded dyld images
// as Linux /proc/<pid>/maps style lines, one line per mapped segment.
void write_process_maps_file();

// Register a _dyld add-image callback so late dlopen() images are reflected
// in the maps file without any polling thread.
void register_image_add_callback();

// One stack-only, allocation-free probe for the global SHM segment. Returns
// true when the host has already created it; false (quietly) otherwise, with
// host errno preserved. Never retries and never blocks.
bool global_shm_ready_once();

} // namespace agent_darwin
} // namespace bpftime

#endif // BPFTIME_DARWIN_AGENT_HPP
