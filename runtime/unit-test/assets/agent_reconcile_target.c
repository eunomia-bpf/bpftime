/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */

/* Tiny DSO loaded late (via dlopen) by agent_reconcile_helper. The uprobe
 * registered against it exists in SHM before the DSO is mapped, so the agent
 * must reconcile after the load.
 */
int bpftime_reconcile_target_func(int x)
{
	return x + 1;
}
