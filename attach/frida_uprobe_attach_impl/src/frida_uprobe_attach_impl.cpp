/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "frida_uprobe_attach_impl.hpp"
#include "frida-gum.h"
#include "frida_attach_private_data.hpp"
#include "frida_attach_utils.hpp"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <cerrno>
#include <memory>
#include <typeinfo>
#include <utility>
#include <unistd.h>
#include "frida_internal_attach_entry.hpp"
#include "frida_attach_entry.hpp"

using namespace bpftime::attach;

frida_attach_impl::~frida_attach_impl()
{
	gum_object_unref((GumInterceptor *)interceptor);
}

frida_attach_impl::frida_attach_impl()
{
	SPDLOG_DEBUG("Initializing frida attach manager");
	gum_init_embedded();
	interceptor = gum_interceptor_obtain();
}

int frida_attach_impl::attach_at(void *func_addr, callback_variant &&cb)
{
	auto itr = internal_attaches.find(func_addr);
	if (itr == internal_attaches.end()) {
		// Create a frida attach entry
		itr = internal_attaches
			      .emplace(func_addr,
				       std::make_unique<
					       frida_internal_attach_entry>(
					       func_addr,
					       from_cb_idx_to_attach_type(
						       cb.index()),
					       (GumInterceptor *)interceptor))
			      .first;
		SPDLOG_DEBUG("Created frida attach entry for func addr {:x}",
			     (uintptr_t)func_addr);
	} else if (itr->second->has_override()) {
		SPDLOG_ERROR(
			"Function {} was already attached with replace or filter, cannot attach anything else");
		return -EEXIST;
	}

	auto &inner_attach = itr->second;
	int used_id = this->allocate_id();
	frida_attach_entry ent(used_id, std::move(cb), func_addr);
	int result = ent.self_id;
	auto inserted_attach_entry =
		this->attaches
			.emplace(ent.self_id,
				 std::make_unique<frida_attach_entry>(
					 std::move(ent)))
			.first;
	inner_attach->user_attaches.push_back(
		inserted_attach_entry->second.get());
	inserted_attach_entry->second->internal_attach = inner_attach.get();
	return result;
}

int frida_attach_impl::create_uprobe_at(void *func_addr, uprobe_callback &&cb)
{
	return attach_at(
		func_addr,
		callback_variant(std::in_place_index_t<ATTACH_UPROBE_INDEX>(),
				 cb));
}

int frida_attach_impl::create_uretprobe_at(void *func_addr,
					   uretprobe_callback &&cb)
{
	return attach_at(
		func_addr,
		callback_variant(
			std::in_place_index_t<ATTACH_URETPROBE_INDEX>(), cb));
}

int frida_attach_impl::create_uprobe_override_at(void *func_addr,
						 uprobe_override_callback &&cb)
{
	return attach_at(
		func_addr,
		callback_variant(
			std::in_place_index_t<ATTACH_UPROBE_OVERRIDE_INDEX>(),
			cb));
}

int frida_attach_impl::detach_by_id(int id)
{
	void *drop_func_addr = nullptr;
	if (auto itr = attaches.find(id); itr != attaches.end()) {
		auto p = itr->second->internal_attach;

		auto &user_attaches = p->user_attaches;
		auto tail =
			std::remove_if(user_attaches.begin(),
				       user_attaches.end(),
				       [&](const auto &v) -> bool {
					       return v == itr->second.get();
				       });
		user_attaches.resize(tail - user_attaches.begin());
		attaches.erase(itr);
		if (p->user_attaches.empty()) {
			drop_func_addr = p->function;
		}
	} else {
		SPDLOG_ERROR("Unable to find attach id {}", id);
		errno = -ENOENT;
	}
	if (drop_func_addr)
		internal_attaches.erase(drop_func_addr);
	return 0;
}
void frida_attach_impl::iterate_attaches(attach_iterate_callback cb)
{
	for (const auto &[k, v] : attaches) {
		cb(k, v->function, v->get_type());
	}
}

int frida_attach_impl::detach_by_func_addr(const void *func)
{
	if (auto itr = internal_attaches.find((void *)func);
	    itr != internal_attaches.end()) {
		auto uattaches = itr->second->user_attaches;
		for (auto attach_entry : uattaches) {
			attaches.erase(attach_entry->self_id);
		}
		internal_attaches.erase(itr);
		return 0;
	} else {
		return -ENOENT;
	}
}

int frida_attach_impl::create_attach_with_ebpf_callback(
	ebpf_run_callback &&cb, const attach_private_data &private_data,
	int attach_type)
{
	try {
		auto &sub = dynamic_cast<const frida_attach_private_data &>(
			private_data);

		auto attach_callback = [=](const pt_regs &regs) {
			uint64_t ret;
			if (int err = cb((void *)&regs, sizeof(regs), &ret);
			    err < 0) {
				SPDLOG_ERROR(
					"Failed to run ebpf callback at frida attach manager for attach type {}, err={}",
					attach_type, err);
			}
		};
		if (attach_type == ATTACH_UPROBE) {
			return create_uprobe_at((void *)(uintptr_t)sub.addr,
						std::move(attach_callback));
		} else if (attach_type == ATTACH_URETPROBE) {
			return create_uretprobe_at((void *)(uintptr_t)sub.addr,
						   std::move(attach_callback));
		} else if (attach_type == ATTACH_UPROBE_OVERRIDE) {
			return create_uprobe_override_at(
				(void *)(uintptr_t)sub.addr,
				std::move(attach_callback));
		} else {
			SPDLOG_ERROR(
				"Unsupported attach type by frida attach manager: {}",
				attach_type);
			return -ENOTSUP;
		}
	} catch (const std::bad_cast &ex) {
		SPDLOG_ERROR(
			"Frida attach manager expected a private data of type frida_attach_private_data: {}",
			ex.what());
		return -EINVAL;
	}
}
