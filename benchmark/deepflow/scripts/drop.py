# Calculating the performance drop percentage for each size, using 'no-probe' as the 100% baseline
# The formula used: Performance Drop = ((No-probe - Other) / No-probe) * 100


def calculate_performance_drop(averages, baseline_key):
    performance_drop = {}
    baseline = averages[baseline_key]

    for sub_key in averages:
        if sub_key == baseline_key:
            continue  # Skipping the baseline itself

        performance_drop[sub_key] = {}
        for size in baseline:
            baseline_request = baseline[size]["average_request"]
            baseline_transfer = baseline[size]["average_transfer"]

            current_request = averages[sub_key][size]["average_request"]
            current_transfer = averages[sub_key][size]["average_transfer"]

            request_drop = (
                ((baseline_request - current_request) / baseline_request) * 100
                if baseline_request
                else 0
            )
            transfer_drop = (
                ((baseline_transfer - current_transfer) / baseline_transfer) * 100
                if baseline_transfer
                else 0
            )

            performance_drop[sub_key][size] = {
                "request_drop": request_drop,
                "transfer_drop": transfer_drop,
            }

    return performance_drop


# Calculating performance drop for HTTPS and HTTP
https_performance_drop = calculate_performance_drop(https_averages, "no-probe")
http_performance_drop = calculate_performance_drop(http_averages, "no-probe")

https_performance_drop, http_performance_drop
