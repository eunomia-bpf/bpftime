// SPDX-License-Identifier: MIT
#include <rocprofiler-sdk/intercept_table.h>

static rocprofiler_intercept_library_cb_t callback;
static void *callback_data;
static unsigned registrations;
static bool reject_registration = true;

extern "C" rocprofiler_status_t rocprofiler_at_intercept_table_registration(
	rocprofiler_intercept_library_cb_t cb, int tables, void *data)
{
	if (reject_registration || tables != ROCPROFILER_HSA_TABLE)
		return ROCPROFILER_STATUS_ERROR;
	callback = cb;
	callback_data = data;
	++registrations;
	return ROCPROFILER_STATUS_SUCCESS;
}

extern "C" void allow_registration()
{
	reject_registration = false;
}
extern "C" unsigned registration_count()
{
	return registrations;
}
extern "C" void register_test_tables(void **tables, uint64_t count)
{
	callback(ROCPROFILER_HSA_TABLE, 0, 0, tables, count, callback_data);
}
