#ifndef _SIMPLE_ATTACH_IMPL
#define _SIMPLE_ATTACH_IMPL

#include "attach_private_data.hpp"
#include "base_attach_impl.hpp"
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
namespace bpftime
{
namespace simple_attach
{

/// Callback type for a simple attach impl
/// (Attach argument, Trigger argument, callback to execute the eBPF program
/// related to this attach) -> status code
using callback_type =
	std::function<int(const std::string &, const std::string &,
			  const attach::ebpf_run_callback &)>;
/// This is a wrapper for base_attach_impl, used together with other helpers in
/// bpf_attach_ctx, to provide simpler but less flexible ways to register a
/// custom attach impl
/// Note that each instance of simple_attach_impl can only hold one attach at
/// the same time.
class simple_attach_impl : public attach::base_attach_impl {
    public:
	int create_attach_with_ebpf_callback(
		attach::ebpf_run_callback &&cb,
		const attach::attach_private_data &private_data,
		int attach_type);
	/// Trigger the event handled by this attach impl
	int trigger(const std::string &);
	int detach_by_id(int id);
	bool is_attached() const
	{
		return usable_id != -1;
	}

	simple_attach_impl(callback_type &&callback, int attach_type)
		: callback(callback), predefined_attach_type(attach_type)
	{
	}

    private:
	/// A callback provided by user, will be triggered if this attach type
	/// was triggered
	callback_type callback;
	/// Specified attach type for this simple_attach_impl. It will be
	/// checked when being attached.
	int predefined_attach_type;

	attach::ebpf_run_callback ebpf_callback;
	int usable_id = -1;
	std::string argument;
};

struct simple_attach_private_data : public attach::attach_private_data {
	std::string data;
	simple_attach_private_data(const std::string_view &data) : data(data)
	{
	}
	int initialize_from_string(const std::string_view &sv) override
	{
		data = sv;
		return 0;
	}
};

// Replace concept with template metaprogramming for older compilers
template <typename T>
struct has_register_attach_impl_helper {
private:
    template <typename C>
    static auto test(int) -> decltype(
        std::declval<C>().register_attach_impl(
            std::declval<std::initializer_list<int>>(),
            std::declval<std::unique_ptr<attach::base_attach_impl>>(),
            std::declval<std::function<std::unique_ptr<attach::attach_private_data>(
                const std::string_view &, int &)>>()
        ), std::true_type());
    
    template <typename>
    static std::false_type test(...);

public:
    using type = decltype(test<T>(0));
};

template <typename T>
struct has_register_attach_impl : has_register_attach_impl_helper<T>::type {};

/// Adds an simple defined attach impl to the specified bpf_attach_ctx
/// A simple attach impl consists up of a callback, an attach type integer, and
/// a trigger. The callback will be called when the trigger function was called.
/// The callback function will receive another callback to execute the desired
/// eBPF program linked with this attach. It will also received the argument
/// provided at attach, and the argument provided at trigger
template <class T>
static inline
typename std::enable_if<has_register_attach_impl<T>::value, std::function<int(const std::string &)>>::type
add_simple_attach_impl_to_attach_ctx(int attach_type, callback_type &&callback,
				     T &attach_ctx)
{
	auto simple_attach_impl = std::make_unique<class simple_attach_impl>(
		std::move(callback), attach_type);
	auto impl_ptr = simple_attach_impl.get();
	auto result = [=](const std::string &s) -> int {
		return impl_ptr->trigger(s);
	};

	attach_ctx.register_attach_impl(
		{ attach_type }, std::move(simple_attach_impl),
		[](const std::string_view &sv, int &err) {
			std::unique_ptr<attach::attach_private_data> priv_data =
				std::make_unique<simple_attach_private_data>(
					sv);
			err = 0;
			return priv_data;
		});
	return result;
}

} // namespace simple_attach
} // namespace bpftime

#endif
