#!/bin/bash

mig_instances=(MIG-62f26659-16c9-5d92-a2fa-4994b2810c3c MIG-f3542379-320f-584f-8b66-35eb2952959e MIG-8f919725-8f02-58dd-95df-c8a074875a52)

for ((idx=0; idx<${#mig_instances[@]}; ++idx)); do
    echo "$idx" ${mig_instances[idx]}
    CUDA_VISIBLE_DEVICES=${mig_instances[idx]} python mig-test.py -i $idx &
    pids[$idx]=$!
done

# wait
for pid in ${pids[*]}; do
    wait $pid
done

