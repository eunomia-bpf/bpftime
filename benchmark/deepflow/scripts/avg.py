# Function to calculate the averages for 'request' and 'transfer' for each size
def calculate_averages(data):
    averages = {}
    for sub_key in data:
        size_sum = {}
        size_count = {}

        # Process each list of entries
        for entry_list in data[sub_key]['details']:
            for item in entry_list:
                size = item['size']
                request = item['request']
                transfer = item['transfer']

                if size not in size_sum:
                    size_sum[size] = {'request': 0, 'transfer': 0}
                    size_count[size] = 0

                size_sum[size]['request'] += request
                size_sum[size]['transfer'] += transfer
                size_count[size] += 1

        # Calculate averages
        averages[sub_key] = {}
        for size in size_sum:
            averages[sub_key][size] = {
                'average_request': size_sum[size]['request'] / size_count[size],
                'average_transfer': size_sum[size]['transfer'] / size_count[size]
            }

    return averages

# Calculating averages for 'https' and 'http'
https_averages = calculate_averages(https_structure)
http_averages = calculate_averages(http_structure)

https_averages, http_averages

# Adjusting the function to handle potential issues with the data and confidence interval calculation
def calculate_detailed_statistics_adjusted(data):
    statistics = {}
    for sub_key in data:
        all_data = {}

        # Collecting all data points for each size
        for entry_list in data[sub_key]['details']:
            for item in entry_list:
                size = item['size']
                request = item['request']
                transfer = item['transfer']

                if size not in all_data:
                    all_data[size] = {'request': [], 'transfer': []}

                all_data[size]['request'].append(request)
                all_data[size]['transfer'].append(transfer)

        # Calculating statistics for each size
        statistics[sub_key] = {}
        for size in all_data:
            request_data = all_data[size]['request']
            transfer_data = all_data[size]['transfer']

            # Median, variance, and extrema
            median_request = np.median(request_data)
            median_transfer = np.median(transfer_data)
            variance_request = np.var(request_data, ddof=1)
            variance_transfer = np.var(transfer_data, ddof=1)
            min_request, max_request = min(request_data), max(request_data)
            min_transfer, max_transfer = min(transfer_data), max(transfer_data)

            # Bootstrap for confidence intervals, handling cases with few data points
            try:
                ci_request = bootstrap((np.array(request_data),), np.median, confidence_level=0.95).confidence_interval
                ci_transfer = bootstrap((np.array(transfer_data),), np.median, confidence_level=0.95).confidence_interval
            except ValueError:
                # Default to None if confidence interval can't be calculated
                ci_request = (None, None)
                ci_transfer = (None, None)

            statistics[sub_key][size] = {
                'median_request': median_request,
                'median_transfer': median_transfer,
                'variance_request': variance_request,
                'variance_transfer': variance_transfer,
                'min_request': min_request,
                'max_request': max_request,
                'min_transfer': min_transfer,
                'max_transfer': max_transfer,
                'ci_request': ci_request,
                'ci_transfer': ci_transfer
            }

    return statistics

# Re-calculating detailed statistics for 'https' and 'http'
https_statistics_adjusted = calculate_detailed_statistics_adjusted(https_structure)
http_statistics_adjusted = calculate_detailed_statistics_adjusted(http_structure)

https_statistics_adjusted, http_statistics_adjusted

