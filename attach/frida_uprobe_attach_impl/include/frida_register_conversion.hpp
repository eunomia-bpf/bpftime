/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _ATTACH_INTERNAL_HPP
#define _ATTACH_INTERNAL_HPP
#include <frida_register_def.hpp>
namespace bpftime
{

#if defined(__x86_64__) || defined(_M_X64)

void convert_gum_cpu_context_to_pt_regs(const struct _GumX64CpuContext &context,
					pt_regs &regs);

void convert_pt_regs_to_gum_cpu_context(const pt_regs &regs,
					struct _GumX64CpuContext &context);

#elif defined(__aarch64__) || defined(_M_ARM64)
void convert_gum_cpu_context_to_pt_regs(
	const struct _GumArm64CpuContext &context, pt_regs &regs);
void convert_pt_regs_to_gum_cpu_context(const pt_regs &regs,
					struct _GumArm64CpuContext &context);
#elif defined(__arm__) || defined(_M_ARM)
void convert_gum_cpu_context_to_pt_regs(const struct _GumArmCpuContext &context,
					pt_regs &regs);

void convert_pt_regs_to_gum_cpu_context(const pt_regs &regs,
					struct _GumArmCpuContext &context);
#else
#error "Unsupported architecture"
#endif
// GType uprobe_listener_get_type();
} // namespace bpftime
#endif
