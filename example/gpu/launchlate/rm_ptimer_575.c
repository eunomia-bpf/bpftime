/* SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause) */
#define _GNU_SOURCE
#include "rm_ptimer_575.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/ioctl.h>
#include <stddef.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#define NV_IOCTL_MAGIC 'F'
#define NV_IOCTL_BASE 200
#define NV_ESC_IOCTL_XFER_CMD (NV_IOCTL_BASE + 11)
#define NV_ESC_RM_FREE 0x29U
#define NV_ESC_RM_CONTROL 0x2aU
#define NV_ESC_RM_ALLOC 0x2bU
#define NV01_ROOT_CLIENT 0x00000041U
#define NV01_DEVICE_0 0x00000080U
#define NV20_SUBDEVICE_0 0x00002080U
#define RM_ENDPOINTS_V1_COMMAND 0x20800408U

typedef struct {
	uint32_t cmd;
	uint32_t size;
	void *ptr __attribute__((aligned(8)));
} nv_ioctl_xfer_t;

typedef struct {
	uint32_t hRoot;
	uint32_t hObjectParent;
	uint32_t hObjectNew;
	uint32_t hClass;
	void *pAllocParms __attribute__((aligned(8)));
	void *pRightsRequested __attribute__((aligned(8)));
	uint32_t paramsSize;
	uint32_t flags;
	uint32_t status;
} nvos64_parameters;

typedef struct {
	uint32_t hRoot;
	uint32_t hObjectParent;
	uint32_t hObjectOld;
	uint32_t status;
} nvos00_parameters;

typedef struct {
	uint32_t hClient;
	uint32_t hObject;
	uint32_t cmd;
	uint32_t flags;
	void *params __attribute__((aligned(8)));
	uint32_t paramsSize;
	uint32_t status;
} nvos54_parameters;

typedef struct {
	uint32_t deviceId;
	uint32_t hClientShare;
	uint32_t hTargetClient;
	uint32_t hTargetDevice;
	uint32_t flags;
	uint64_t vaSpaceSize __attribute__((aligned(8)));
	uint64_t vaStartInternal __attribute__((aligned(8)));
	uint64_t vaLimitInternal __attribute__((aligned(8)));
	uint32_t vaMode;
} nv0080_alloc_parameters;

typedef struct {
	uint32_t subDeviceId;
} nv2080_alloc_parameters;

typedef struct {
	uint64_t cpuBeforeNs __attribute__((aligned(8)));
	uint64_t gpuTimeNs __attribute__((aligned(8)));
	uint64_t cpuAfterNs __attribute__((aligned(8)));
} rm_endpoint_parameters;

_Static_assert(sizeof(nv_ioctl_xfer_t) == 16, "575 xfer ABI");
_Static_assert(sizeof(nvos64_parameters) == 48, "575 alloc ABI");
_Static_assert(offsetof(nvos64_parameters, status) == 40, "575 alloc status");
_Static_assert(sizeof(nvos00_parameters) == 16, "575 free ABI");
_Static_assert(sizeof(nvos54_parameters) == 32, "575 control ABI");
_Static_assert(offsetof(nvos54_parameters, params) == 16,
	       "575 control params");
_Static_assert(sizeof(nv0080_alloc_parameters) == 56, "575 device ABI");
_Static_assert(sizeof(nv2080_alloc_parameters) == 4, "575 subdevice ABI");
_Static_assert(sizeof(rm_endpoint_parameters) == 24, "endpoint-v1 ABI");

static int raw_ns(uint64_t *value)
{
	struct timespec now;

	if (!value || clock_gettime(CLOCK_MONOTONIC_RAW, &now) != 0)
		return -errno;
	if (now.tv_sec < 0 || now.tv_nsec < 0 || now.tv_nsec >= 1000000000L ||
	    (uint64_t)now.tv_sec > UINT64_MAX / 1000000000ULL)
		return -ERANGE;
	*value = (uint64_t)now.tv_sec * 1000000000ULL + (uint64_t)now.tv_nsec;
	return 0;
}

static int xfer(int fd, uint32_t command, void *payload, uint32_t size)
{
	nv_ioctl_xfer_t args = { .cmd = command, .size = size, .ptr = payload };
	unsigned long request = _IOWR(NV_IOCTL_MAGIC, NV_ESC_IOCTL_XFER_CMD,
				      nv_ioctl_xfer_t);

	return ioctl(fd, request, &args) < 0 ? -errno : 0;
}

static int alloc_object(int fd, uint32_t root, uint32_t parent,
			uint32_t class_id, void *params, uint32_t params_size,
			uint32_t *object)
{
	nvos64_parameters args = {
		.hRoot = root,
		.hObjectParent = parent,
		.hClass = class_id,
		.pAllocParms = params,
		.paramsSize = params_size,
	};
	int err = xfer(fd, NV_ESC_RM_ALLOC, &args, sizeof(args));

	if (err)
		return err;
	if (args.status || !args.hObjectNew)
		return -EREMOTEIO;
	*object = args.hObjectNew;
	return 0;
}

static int free_root(int fd, uint32_t root)
{
	nvos00_parameters args = {
		.hRoot = root,
		.hObjectParent = root,
		.hObjectOld = root,
	};
	int err = xfer(fd, NV_ESC_RM_FREE, &args, sizeof(args));

	if (err)
		return err;
	return args.status ? -EREMOTEIO : 0;
}

static int checked_offset(uint64_t gpu, uint64_t cpu, int64_t padding,
			  int64_t *result)
{
	__int128 value = (__int128)gpu - (__int128)cpu + padding;

	if (value < INT64_MIN || value > INT64_MAX)
		return -ERANGE;
	*result = (int64_t)value;
	return 0;
}

static int derive_sample(uint64_t outer_before, uint64_t outer_after,
			 uint64_t cpu_before, uint64_t gpu,
			 uint64_t cpu_after,
			 struct rm_ptimer_575_sample *sample)
{
	if (!sample || !outer_before || outer_after < outer_before ||
	    cpu_before < outer_before || cpu_after < cpu_before ||
	    cpu_after > outer_after || !gpu ||
	    outer_after - outer_before >= RM_PTIMER_MAX_OUTER_NS)
		return -ERANGE;
	memset(sample, 0, sizeof(*sample));
	sample->outer_before_raw_ns = outer_before;
	sample->outer_after_raw_ns = outer_after;
	sample->cpu_before_raw_ns = cpu_before;
	sample->gpu_ptimer_ns = gpu;
	sample->cpu_after_raw_ns = cpu_after;
	sample->outer_width_ns = outer_after - outer_before;
	sample->selected_gap_ns = cpu_after - cpu_before;
	if (checked_offset(gpu, cpu_after, -(int64_t)RM_PTIMER_QUANTIZATION_NS,
			   &sample->offset_low_ns) ||
	    checked_offset(gpu, cpu_before, (int64_t)RM_PTIMER_QUANTIZATION_NS,
			   &sample->offset_high_ns) ||
	    sample->offset_high_ns < sample->offset_low_ns)
		return -ERANGE;
	sample->bracket_width_ns = (uint64_t)sample->offset_high_ns -
				   (uint64_t)sample->offset_low_ns;
	return 0;
}

void rm_ptimer_575_client_init(struct rm_ptimer_575_client *client)
{
	if (!client)
		return;
	memset(client, 0, sizeof(*client));
	client->control_fd = -1;
	client->gpu_fd = -1;
}

int rm_ptimer_575_open(struct rm_ptimer_575_client *client)
{
	nv0080_alloc_parameters device_params = { .deviceId = 0 };
	nv2080_alloc_parameters subdevice_params = { .subDeviceId = 0 };
	int err;

	if (!client)
		return -EINVAL;
	rm_ptimer_575_client_init(client);
	client->control_fd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
	if (client->control_fd < 0)
		return -errno;
	client->gpu_fd = open("/dev/nvidia0", O_RDWR | O_CLOEXEC);
	if (client->gpu_fd < 0) {
		err = -errno;
		goto fail;
	}
	err = alloc_object(client->control_fd, 0, 0, NV01_ROOT_CLIENT, NULL, 0,
			   &client->root);
	if (err)
		goto fail;
	device_params.hClientShare = client->root;
	err = alloc_object(client->control_fd, client->root, client->root,
			   NV01_DEVICE_0, &device_params, sizeof(device_params),
			   &client->device);
	if (err)
		goto fail;
	err = alloc_object(client->control_fd, client->root, client->device,
			   NV20_SUBDEVICE_0, &subdevice_params,
			   sizeof(subdevice_params), &client->subdevice);
	if (err)
		goto fail;
	return 0;

fail:
	(void)rm_ptimer_575_close(client);
	return err;
}

int rm_ptimer_575_sample(struct rm_ptimer_575_client *client,
			 struct rm_ptimer_575_sample *sample)
{
	rm_endpoint_parameters endpoints = {0};
	nvos54_parameters control;
	uint64_t before, after;
	unsigned long request = _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_CONTROL,
				      nvos54_parameters);
	int err;

	if (!client || client->control_fd < 0 || !client->root ||
	    !client->subdevice || !sample)
		return -EINVAL;
	memset(&control, 0, sizeof(control));
	control.hClient = client->root;
	control.hObject = client->subdevice;
	control.cmd = RM_ENDPOINTS_V1_COMMAND;
	control.params = &endpoints;
	control.paramsSize = sizeof(endpoints);
	if ((err = raw_ns(&before)))
		return err;
	if (ioctl(client->control_fd, request, &control) < 0)
		return -errno;
	if ((err = raw_ns(&after)))
		return err;
	if (control.status)
		return -EREMOTEIO;
	err = derive_sample(before, after, endpoints.cpuBeforeNs,
			    endpoints.gpuTimeNs, endpoints.cpuAfterNs, sample);
	if (!err)
		sample->rm_status = control.status;
	return err;
}

int rm_ptimer_575_close(struct rm_ptimer_575_client *client)
{
	int result = 0;

	if (!client)
		return -EINVAL;
	if (client->root && client->control_fd >= 0)
		result = free_root(client->control_fd, client->root);
	if (client->gpu_fd >= 0 && close(client->gpu_fd) && !result)
		result = -errno;
	if (client->control_fd >= 0 && close(client->control_fd) && !result)
		result = -errno;
	rm_ptimer_575_client_init(client);
	return result;
}

int rm_ptimer_575_self_test(void)
{
	struct rm_ptimer_575_sample sample;

	if (derive_sample(900, 1300, 1000, 2000, 1100, &sample) ||
	    sample.outer_width_ns != 400 || sample.selected_gap_ns != 100 ||
	    sample.offset_low_ns != 868 || sample.offset_high_ns != 1032 ||
	    sample.bracket_width_ns != 164)
		return 1;
	if (derive_sample(900, 1300, 1000, 500, 1101, &sample) ||
	    sample.offset_low_ns != -633 || sample.offset_high_ns != -468 ||
	    sample.bracket_width_ns != 165)
		return 1;
	if (derive_sample(1001, 1300, 1000, 2000, 1100, &sample) != -ERANGE ||
	    derive_sample(900, 1099, 1000, 2000, 1100, &sample) != -ERANGE ||
	    derive_sample(1300, 900, 1000, 2000, 1100, &sample) != -ERANGE ||
	    derive_sample(900, 1300, 1100, 2000, 1000, &sample) != -ERANGE ||
	    derive_sample(900, 900 + RM_PTIMER_MAX_OUTER_NS, 1000, 2000,
			  1100, &sample) != -ERANGE ||
	    checked_offset(UINT64_MAX, 0, 0, &sample.offset_low_ns) != -ERANGE)
		return 1;
	return 0;
}
