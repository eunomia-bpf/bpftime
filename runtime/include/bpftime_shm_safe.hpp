/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _BPFTIME_SHM_SAFE_HPP
#define _BPFTIME_SHM_SAFE_HPP

#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace bpftime
{

class shm_init_ctx;

namespace detail
{

template <typename T> struct is_forbidden_shm_std_type : std::false_type {};

template <typename CharT, typename Traits, typename Alloc>
struct is_forbidden_shm_std_type<std::basic_string<CharT, Traits, Alloc>>
	: std::true_type {};

template <typename T, typename Alloc>
struct is_forbidden_shm_std_type<std::vector<T, Alloc>> : std::true_type {};

template <typename K, typename V, typename Cmp, typename Alloc>
struct is_forbidden_shm_std_type<std::map<K, V, Cmp, Alloc>>
	: std::true_type {};

template <typename K, typename V, typename Cmp, typename Alloc>
struct is_forbidden_shm_std_type<std::multimap<K, V, Cmp, Alloc>>
	: std::true_type {};

template <typename K, typename V, typename Hash, typename Eq, typename Alloc>
struct is_forbidden_shm_std_type<std::unordered_map<K, V, Hash, Eq, Alloc>>
	: std::true_type {};

template <typename K, typename V, typename Hash, typename Eq, typename Alloc>
struct is_forbidden_shm_std_type<std::unordered_multimap<K, V, Hash, Eq, Alloc>>
	: std::true_type {};

template <typename K, typename Cmp, typename Alloc>
struct is_forbidden_shm_std_type<std::set<K, Cmp, Alloc>> : std::true_type {};

template <typename K, typename Hash, typename Eq, typename Alloc>
struct is_forbidden_shm_std_type<std::unordered_set<K, Hash, Eq, Alloc>>
	: std::true_type {};

template <typename Sig>
struct is_forbidden_shm_std_type<std::function<Sig>> : std::true_type {};

template <typename T, typename Deleter>
struct is_forbidden_shm_std_type<std::unique_ptr<T, Deleter>>
	: std::true_type {};

template <typename T>
struct is_forbidden_shm_std_type<std::shared_ptr<T>> : std::true_type {};

template <typename T>
struct is_forbidden_shm_std_type<std::weak_ptr<T>> : std::true_type {};

template <> struct is_forbidden_shm_std_type<std::mutex> : std::true_type {};
template <>
struct is_forbidden_shm_std_type<std::recursive_mutex> : std::true_type {};

template <typename T> struct is_shm_safe_impl;

struct shm_any_field {
	template <typename U> operator U() const noexcept;
};

struct shm_safe_field {
	template <typename U>
	requires(is_shm_safe_impl<U>::value)
	operator U() const noexcept;
};

template <typename T, std::size_t... I>
constexpr bool can_brace_any(std::index_sequence<I...>) noexcept
{
	return requires { T{ (static_cast<void>(I), shm_any_field{})... }; };
}

template <typename T, std::size_t... I>
constexpr bool can_brace_safe(std::index_sequence<I...>) noexcept
{
	return requires { T{ (static_cast<void>(I), shm_safe_field{})... }; };
}

template <typename T, std::size_t N = 8>
constexpr bool small_aggregate_shm_safe() noexcept
{
	if constexpr (N == 0) {
		return true;
	} else if constexpr (can_brace_any<T>(std::make_index_sequence<N>{})) {
		return can_brace_safe<T>(std::make_index_sequence<N>{});
	} else {
		return small_aggregate_shm_safe<T, N - 1>();
	}
}

// Inspects small plain aggregates (up to 8 flattened fields) and rejects any
// aggregate holding C++ reference members (!is_default_constructible_v).
// Non-aggregate classes with private members cannot be reflected in C++20;
// they are covered by the top-level polymorphism/pointer/STL checks and the
// non-copyable shm_init_ctx constructor discipline.
template <typename T> constexpr bool aggregate_shm_safe() noexcept
{
	if constexpr (!std::is_aggregate_v<T> || std::is_union_v<T> ||
		      !std::is_copy_constructible_v<T>) {
		return true;
	} else if constexpr (!std::is_default_constructible_v<T>) {
		return false;
	} else if constexpr (can_brace_any<T>(std::make_index_sequence<9>{})) {
		return true;
	} else {
		return small_aggregate_shm_safe<T>();
	}
}

template <typename T> struct is_shm_safe_compound : std::true_type {};

template <typename E>
struct is_shm_safe_compound<std::optional<E>> : is_shm_safe_impl<E> {};

template <typename A, typename B>
struct is_shm_safe_compound<std::pair<A, B>>
	: std::bool_constant<is_shm_safe_impl<A>::value &&
			     is_shm_safe_impl<B>::value> {};

template <typename... Es>
struct is_shm_safe_compound<std::tuple<Es...>>
	: std::bool_constant<(is_shm_safe_impl<Es>::value && ...)> {};

template <typename... Es>
struct is_shm_safe_compound<std::variant<Es...>>
	: std::bool_constant<(is_shm_safe_impl<Es>::value && ...)> {};

template <typename E, typename Alloc>
struct is_shm_safe_compound<boost::interprocess::vector<E, Alloc>>
	: is_shm_safe_impl<E> {};

template <typename T> struct is_shm_safe_unqualified {
	static constexpr bool value =
		!std::is_pointer_v<T> && !std::is_member_pointer_v<T> &&
		!std::is_reference_v<T> && !std::is_polymorphic_v<T> &&
		!std::is_same_v<T, shm_init_ctx> &&
		!is_forbidden_shm_std_type<T>::value &&
		is_shm_safe_compound<T>::value && aggregate_shm_safe<T>();
};

template <typename E, std::size_t N> struct is_shm_safe_unqualified<E[N]> {
	static constexpr bool value = is_shm_safe_impl<E>::value;
};

template <typename T> struct is_shm_safe_impl {
	static constexpr bool value =
		!std::is_pointer_v<T> && !std::is_reference_v<T> &&
		is_shm_safe_unqualified<std::remove_cv_t<T>>::value;
};

} // namespace detail

// Compile-time gate for types placed inside managed_shared_memory:
// rejects virtual tables (vptr), raw/member pointers, references, shm_init_ctx,
// and process-local STL heap containers or std::mutex instances.
template <typename T>
concept shm_safe_type = detail::is_shm_safe_impl<T>::value;

template <shm_safe_type T, typename Memory, typename NameOrTag,
	  typename... Args>
T *shm_construct(Memory &memory, const NameOrTag &name_or_tag, Args &&...args)
{
	return memory.template construct<T>(name_or_tag)(
		std::forward<Args>(args)...);
}

template <shm_safe_type T, typename Memory, typename... Args>
T *shm_find_or_construct(Memory &memory, const char *name, Args &&...args)
{
	return memory.template find_or_construct<T>(name)(
		std::forward<Args>(args)...);
}

// Non-copyable, non-storable construction context passed to SHM object
// constructors instead of a process-local managed_shared_memory& (#959).
class shm_init_ctx {
    public:
	using segment_manager_t =
		boost::interprocess::managed_shared_memory::segment_manager;

	shm_init_ctx(boost::interprocess::managed_shared_memory &mem) noexcept
		: sm_(mem.get_segment_manager())
	{
	}
	explicit shm_init_ctx(segment_manager_t *sm) noexcept : sm_(sm)
	{
	}

	shm_init_ctx(const shm_init_ctx &) = delete;
	shm_init_ctx &operator=(const shm_init_ctx &) = delete;
	shm_init_ctx(shm_init_ctx &&) = delete;
	shm_init_ctx &operator=(shm_init_ctx &&) = delete;

	segment_manager_t *segment_manager() const noexcept
	{
		return sm_;
	}

	template <typename T>
	boost::interprocess::allocator<T, segment_manager_t>
	allocator() const noexcept
	{
		return boost::interprocess::allocator<T, segment_manager_t>(
			sm_);
	}

    private:
	segment_manager_t *sm_;
};

} // namespace bpftime

#endif // _BPFTIME_SHM_SAFE_HPP
