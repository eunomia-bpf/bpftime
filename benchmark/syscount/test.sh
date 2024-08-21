#!/bin/bash

total_read_time=0
total_sendmsg_time=0

for i in {1..10}
do
    # output=$(sudo AGENT_SO=/tmp/bpftime/runtime/agent/libbpftime-agent.so LD_PRELOAD=/tmp/bpftime/attach/text_segment_transformer/libbpftime-agent-transformer.so ./benchmark/syscount/read-sendmsg)
    output=$(sudo ./benchmark/syscount/read-sendmsg)
    read_time=$(echo "$output" | grep "Average read() time" | awk '{print $4}')
    sendmsg_time=$(echo "$output" | grep "Average sendmsg() time" | awk '{print $4}')
    
    total_read_time=$((total_read_time + read_time))
    total_sendmsg_time=$((total_sendmsg_time + sendmsg_time))
done

avg_read_time=$((total_read_time / 10))
avg_sendmsg_time=$((total_sendmsg_time / 10))

echo "Average read() time over 10 runs: $avg_read_time ns"
echo "Average sendmsg() time over 10 runs: $avg_sendmsg_time ns"
