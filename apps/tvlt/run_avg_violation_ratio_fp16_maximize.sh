#!/bin/bash

mkdir violation_ratio_results
touch violation_ratio_results/fp16_maximize_3.txt
> violation_ratio_results/fp16_maximize_3.txt
max_acc=0.743
diff=0.03
min_acc=0$(echo "($max_acc-$diff)"| bc -l)

# running fp16 with no policy change
for qps in {10..120..10}
do
    python test_violation_ratio.py --float_format fp16 --avg_violation_ratio_file violation_ratio_results/fp16_maximize_3.txt --policy maximize --fixed_qps $qps --min_acc $min_acc --max_acc $max_acc  --trace_file ./tmp/bogus.txt --profiled_results mosei_batch_profiling_result/mosei_batch_fp16.json --pkl mosei_b_to_a/b_to_a_fp16.pkl
done


touch violation_ratio_results/fp16_maximize_6.txt
> violation_ratio_results/fp16_maximize_6.txt
max_acc=0.743
diff=0.06
min_acc=0$(echo "($max_acc-$diff)"| bc -l)

# running fp16 with no policy change
for qps in {10..120..10}
do
    python test_violation_ratio.py --float_format fp16 --avg_violation_ratio_file violation_ratio_results/fp16_maximize_6.txt --policy maximize --fixed_qps $qps --min_acc $min_acc --max_acc $max_acc  --trace_file ./tmp/bogus.txt --profiled_results mosei_batch_profiling_result/mosei_batch_fp16.json --pkl mosei_b_to_a/b_to_a_fp16.pkl
done


touch violation_ratio_results/fp16_maximize_9.txt
> violation_ratio_results/fp16_maximize_9.txt
max_acc=0.743
diff=0.09
min_acc=0$(echo "($max_acc-$diff)"| bc -l)

# running fp16 with no policy change
for qps in {10..120..10}
do
    python test_violation_ratio.py --float_format fp16 --avg_violation_ratio_file violation_ratio_results/fp16_maximize_9.txt --policy maximize --fixed_qps $qps --min_acc $min_acc --max_acc $max_acc  --trace_file ./tmp/bogus.txt --profiled_results mosei_batch_profiling_result/mosei_batch_fp16.json --pkl mosei_b_to_a/b_to_a_fp16.pkl
done

touch violation_ratio_results/fp16_maximize_12.txt
> violation_ratio_results/fp16_maximize_12.txt
max_acc=0.743
diff=0.12
min_acc=0$(echo "($max_acc-$diff)"| bc -l)

# running fp16 with no policy change
for qps in {10..120..10}
do
    python test_violation_ratio.py --float_format fp16 --avg_violation_ratio_file violation_ratio_results/fp16_maximize_12.txt --policy maximize --fixed_qps $qps --min_acc $min_acc --max_acc $max_acc  --trace_file ./tmp/bogus.txt --profiled_results mosei_batch_profiling_result/mosei_batch_fp16.json --pkl mosei_b_to_a/b_to_a_fp16.pkl
done