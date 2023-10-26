#include <handler/handler_manager.hpp>
namespace bpftime
{
handler_manager::handler_manager(managed_shared_memory &mem,
				 size_t max_fd_count)
	: handlers(max_fd_count,
		   handler_variant_allocator(mem.get_segment_manager()))
{
}

handler_manager::~handler_manager()
{
	for (std::size_t i = 0; i < handlers.size(); i++) {
		assert(!is_allocated(i));
	}
}

const handler_variant &handler_manager::get_handler(int fd) const
{
	return handlers[fd];
}

const handler_variant &handler_manager::operator[](int idx) const
{
	return handlers[idx];
}
std::size_t handler_manager::size() const
{
	return handlers.size();
}

int handler_manager::set_handler(int fd, handler_variant &&handler,
				  managed_shared_memory &memory)
{
	if (is_allocated(fd)) {
		spdlog::error("set_handler failed for fd {} aleady exists", fd);
		return -ENOENT;
	}
	handlers[fd] = std::move(handler);
	if (std::holds_alternative<bpf_map_handler>(handlers[fd])) {
		std::get<bpf_map_handler>(handlers[fd]).map_init(memory);
	}
	return fd;
}

bool handler_manager::is_allocated(int fd) const
{
	if (fd < 0 || (std::size_t)fd >= handlers.size()) {
		return false;
	}
	return !std::holds_alternative<unused_handler>(handlers.at(fd));
}

void handler_manager::clear_fd_at(int fd, managed_shared_memory &memory)
{
	if (fd < 0 || (std::size_t)fd >= handlers.size()) {
		return;
	}
	if (std::holds_alternative<bpf_map_handler>(handlers[fd])) {
		std::get<bpf_map_handler>(handlers[fd]).map_free(memory);
	}
	handlers[fd] = unused_handler();
}

int handler_manager::find_minimal_unused_idx() const {
	for (std::size_t i = 0; i < handlers.size(); i++) {
		if (!is_allocated(i)) {
			return i;
		}
	}
	return -1;
}

void handler_manager::clear_all(managed_shared_memory &memory)
{
	for (std::size_t i = 0; i < handlers.size(); i++) {
		if (is_allocated(i)) {
			clear_fd_at(i, memory);
		}
	}
}

} // namespace bpftime
