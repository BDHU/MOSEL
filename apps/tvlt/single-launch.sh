#!/bin/bash

device=0

for ((idx=0; idx<3; ++idx)); do
    CUDA_VISIBLE_DEVICES=${device} python mig-test.py -i $idx &
    pids[$idx]=$!
done

# wait
for pid in ${pids[*]}; do
    wait $pid
done

