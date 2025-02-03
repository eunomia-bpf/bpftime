#include "./lru_var_hash_map.hpp"
#include "spdlog/spdlog.h"
#include <boost/interprocess/detail/segment_manager_helper.hpp>
#include <boost/interprocess/smart_ptr/shared_ptr.hpp>
#include "linux/bpf.h"
using namespace bpftime;

static bool is_good_update_flag(uint64_t flags)
{
	return flags == BPF_ANY || flags == BPF_EXIST || flags == BPF_NOEXIST;
}

lru_var_hash_map_impl::lru_var_hash_map_impl(
	boost::interprocess::managed_shared_memory &memory, size_t key_size,
	size_t value_size, size_t max_entries)
	: map_impl(max_entries, memory.get_segment_manager()),
	  key_size(key_size), value_size(value_size), max_entries(max_entries),
	  key_vec(key_size, memory.get_segment_manager()),
	  value_vec(value_size, memory.get_segment_manager()), memory(memory)

{
}

void *lru_var_hash_map_impl::elem_lookup(const void *key)
{
	SPDLOG_TRACE("Peform elem lookup of lru var hash map");
	key_vec.assign((uint8_t *)key, (uint8_t *)key + key_size);
	if (auto itr = map_impl.find(key_vec); itr != map_impl.end()) {
		SPDLOG_TRACE("Exit elem lookup of lru var hash map");
		move_to_head(itr->second.linked_list_entry);
		return &itr->second.value[0];

	} else {
		SPDLOG_TRACE("Exit elem lookup of lru var hash map");
		errno = ENOENT;
		return nullptr;
	}
}

long lru_var_hash_map_impl::elem_update(const void *key, const void *value,
					uint64_t flags)
{
	// Check flags
	if (!is_good_update_flag(flags)) {
		errno = EINVAL;
		return -1;
	}
	key_vec.assign((uint8_t *)key, (uint8_t *)key + key_size);
	auto itr = map_impl.find(key_vec);
	bool element_exists = itr != map_impl.end();

	if (flags == BPF_NOEXIST && element_exists) {
		errno = EEXIST;
		return -1;
	}
	if (flags == BPF_EXIST && !element_exists) {
		errno = ENOENT;
		return -1;
	}
	if (element_exists == false && map_impl.size() == max_entries) {
		// Evict least recently used entry
		SPDLOG_DEBUG("Evicting least recently used entry");
		map_impl.erase(lru_link_list_tail->key);
		evict_entry(lru_link_list_tail);
	}
	if (element_exists) {
		// Update the value
		itr->second.value.assign((uint8_t *)value,
					 (uint8_t *)value + value_size);
		move_to_head(itr->second.linked_list_entry);
	} else {
		// Insert a new one
		value_vec.assign((uint8_t *)value,
				 (uint8_t *)value + value_size);
		auto entry_ptr = insert_new_entry(key_vec);
		hash_map_value value{ .value = value_vec,
				      .linked_list_entry = entry_ptr };
		map_impl.emplace(bi_map_value_ty(key_vec, value));
	}

	return 0;
}

long lru_var_hash_map_impl::elem_delete(const void *key)
{
	key_vec.assign((uint8_t *)key, (uint8_t *)key + key_size);
	auto itr = map_impl.find(key_vec);
	if (itr == map_impl.end()) {
		errno = ENOENT;
		return -1;
	}
	evict_entry(itr->second.linked_list_entry);
	map_impl.erase(itr);
	return 0;
}

int lru_var_hash_map_impl::map_get_next_key(const void *key, void *next_key)
{
	if (key == nullptr) {
		// nullptr means the first key
		auto itr = map_impl.begin();
		if (itr == map_impl.end()) {
			errno = ENOENT;
			return -1;
		}
		std::copy(itr->first.begin(), itr->first.end(),
			  (uint8_t *)next_key);
		return 0;
	}
	// Since we use lock here, we don't need to allocate key_vec and
	// value_vec
	key_vec.assign((uint8_t *)key, (uint8_t *)key + key_size);

	auto itr = map_impl.find(key_vec);
	if (itr == map_impl.end()) {
		// not found, should be refer to the first key
		return map_get_next_key(nullptr, next_key);
	}
	itr++;
	if (itr == map_impl.end()) {
		// If *key* is the last element, returns -1 and *errno*
		// is set to **ENOENT**.
		errno = ENOENT;
		return -1;
	}
	std::copy(itr->first.begin(), itr->first.end(), (uint8_t *)next_key);
	return 0;
}

void lru_var_hash_map_impl::move_to_head(lru_linklist_entry_shared_ptr entry)
{
	// Case1: entry is head
	if (entry == lru_link_list_head)
		return;
	// Case2: entry is not head
	auto prev = entry->previous_entry.lock();
	auto next = entry->next_entry;

	prev->next_entry = next;
	// entry is not tail
	if (next) {
		next->previous_entry = prev;
	}
	// entry is tail
	else {
		lru_link_list_tail = prev;
	}

	entry->next_entry = lru_link_list_head;
	entry->previous_entry.reset();

	lru_link_list_head->previous_entry = entry;
	lru_link_list_head = entry;
}

lru_linklist_entry_shared_ptr
lru_var_hash_map_impl::insert_new_entry(const bytes_vec &key)
{
	auto entry_ptr = boost::interprocess::make_managed_shared_ptr(
		memory.construct<lru_linklist_entry>(
			boost::interprocess::anonymous_instance)(key),
		memory);
	// case1: empty
	if (!lru_link_list_head) {
		lru_link_list_head = lru_link_list_tail = entry_ptr;
	} else {
		entry_ptr->next_entry = lru_link_list_head;
		lru_link_list_head->previous_entry = entry_ptr;
		lru_link_list_head = entry_ptr;
	}
	return entry_ptr;
}

void lru_var_hash_map_impl::evict_entry(lru_linklist_entry_shared_ptr entry)
{
	// case1: entry is tail
	if (entry == lru_link_list_tail) {
		// case 1.1: entry is the only element
		if (entry == lru_link_list_head) {
			lru_link_list_head.reset();
			lru_link_list_tail.reset();
		}
		// case 1.2: entry is not the only element
		else {
			auto prev = entry->previous_entry.lock();
			prev->next_entry.reset();
			lru_link_list_tail = prev;
		}
	}
	// case 2: entry is not tail
	else {
		/// Here it ensures there is at least two elements in the linked
		/// list case 2.1: entry is the head
		if (entry == lru_link_list_head) {
			entry->next_entry->previous_entry.reset();
			lru_link_list_head = entry->next_entry;
		}
		/// case 2.2: entry is in the middle of the list
		else {
			auto prev = entry->previous_entry.lock();
			auto next = entry->next_entry;
			prev->next_entry = next;
			next->previous_entry = prev;
		}
	}
}
