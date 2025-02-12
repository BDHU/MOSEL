#!/bin/bash

mkdir violation_ratio_results
touch violation_ratio_results/fp16_no_mod.txt
> violation_ratio_results/fp16_no_mod.txt

# running fp16 with no policy change
for qps in {10..80..10}
do
    python test_violation_ratio.py --float_format fp16 --avg_violation_ratio_file violation_ratio_results/fp16_no_mod.txt --policy no_policy --fixed_qps $qps --min_acc 0.6 --max_acc 0.743  --trace_file ./tmp/bogus.txt --profiled_results mosei_batch_profiling_result/mosei_batch_fp16.json --pkl mosei_b_to_a/b_to_a_fp16.pkl
done