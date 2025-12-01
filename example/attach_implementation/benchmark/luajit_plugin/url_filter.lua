-- URL filter implementation in Lua

-- Global variables to store statistics
local accepted_count = 0
local rejected_count = 0

-- URL prefix to filter (default to "/")
local url_prefix = "/"

-- Buffer for data exchange between host and Lua
local data_buffer = ""

-- Initialize the URL filter with a given prefix
-- @param prefix The URL prefix to filter. If nil, keeps the default
-- @return 0 on success
function initialize(prefix)
    -- If prefix is provided, update the filter prefix
    if prefix ~= nil and prefix ~= "" then
        url_prefix = prefix
    end
    
    -- Reset counters
    accepted_count = 0
    rejected_count = 0
    
    -- Reset buffer
    data_buffer = ""
    
    return 0
end

-- Helper function to check if a string starts with a prefix
local function str_startswith(str, prefix)
    return string.sub(str, 1, string.len(prefix)) == prefix
end

-- Filter a URL based on the configured prefix
-- @param url The URL to filter
-- @return true if the URL starts with the configured prefix, false otherwise
function url_filter(url)
    -- Check if URL starts with the prefix
    local matches = str_startswith(url, url_prefix)
    
    -- Update counters
    if matches then
        accepted_count = accepted_count + 1
    else
        rejected_count = rejected_count + 1
    end
    
    return matches
end

-- Set the data in the buffer
-- @param data The data to store in the buffer
-- @return 0 on success, -1 if error
function set_buffer(data)
    if data == nil then
        return -1
    end
    
    data_buffer = data
    return 0
end

-- Get the data from the buffer
-- @return The buffer data
function get_buffer()
    return data_buffer
end

-- Get the current counters for accepted and rejected URLs
-- @return accepted_count, rejected_count
function get_counters()
    return accepted_count, rejected_count
end 