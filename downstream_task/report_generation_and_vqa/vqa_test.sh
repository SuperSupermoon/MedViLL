#!/bin/bash
model_path=$1
for path_itr in $(find ${model_path} -type d -name "vqa_*" -exec find {} -name "vqa_*" \;)
    do
        for dset_itr in $(find ${path_itr} -type d -name "*_chest" -exec find {} -name "*_chest" \;)
        do
            for itr in $(find ${dset_itr}/* -name "*.bin" -print0 |xargs -r -0 ls -1 -t | head -1);
            do
                echo "chest"
                echo ${itr}
                python $(dirname "$0")/finetune.py --model_recover_path ${itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_eval True --vqa_rad chest
            done
        done
        for dset_itr in $(find ${path_itr} -type d -name "*_abd" -exec find {} -name "*_abd" \;)
        do
            for itr in $(find ${dset_itr}/* -name "*.bin" -print0 |xargs -r -0 ls -1 -t | head -1);
            do
                echo "abd"
                echo ${itr}
                python $(dirname "$0")/finetune.py --model_recover_path ${itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_eval True  --vqa_rad abd
            done

        done
        for dset_itr in $(find ${path_itr} -type d -name "*_head" -exec find {} -name "*_head" \;)
        do            
            for itr in $(find ${dset_itr}/* -name "*.bin" -print0 |xargs -r -0 ls -1 -t | head -1);
            do
                echo "head"
                echo ${itr}
                python $(dirname "$0")/finetune.py --model_recover_path ${itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_eval True --vqa_rad head
            done        
        done
        for dset_itr in $(find ${path_itr} -type d -name "*_all" -exec find {} -name "*_all" \;)
        do
            for itr in $(find ${dset_itr}/* -name "*.bin" -print0 |xargs -r -0 ls -1 -t | head -1);
            do
                echo "all"
                echo ${itr}
                python $(dirname "$0")/finetune.py --model_recover_path ${itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_eval True --vqa_rad all
            done
        done
    done


#Command:
#CUDA_VISIBLE_DEVICES=5 sh downstream_task/report_generation_and_vqa/vqa_test.sh /home/edlab/jhmoon/mimic_mv_real/mimic-cxr

python downstream_task/report_generation_and_vqa/finetune.py --model_recover_path /home/edlab/jhmoon/mimic_mv_real/mimic-cxr/downstream_model/revision/vqa/medvill_seed1004_mimic-cxr_49/model.50.bin --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_eval True --vqa_rad chest
