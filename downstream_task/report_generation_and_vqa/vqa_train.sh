#!/bin/bash
model_path=$1
for path_itr in $(find ${model_path}/* -name pytorch_model.bin);
    do
    # if [ ${path_itr} = /home/data_storage/mimic-cxr/models/pre-train/medvill_seed3333/49/pytorch_model.bin ]; then
    echo "${path_itr} this will be trained!"
    python -m torch.distributed.launch --nproc_per_node=1 --master_port 9872 --use_env downstream_task/report_generation_and_vqa/finetune.py --model_recover_path /home/edlab/jhmoon/mimic_mv_real/mimic-cxr/downstream_model/test_vqa/_mimic-cxr_finetune_only/model.50.bin --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad chest
    # python $(dirname "$0")/finetune.py --model_recover_path ${path_itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad abd
    # python $(dirname "$0")/finetune.py --model_recover_path ${path_itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad head
    python -m torch.distributed.launch --nproc_per_node=1 --use_env $(dirname "$0")/finetune.py --model_recover_path ${path_itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad all
    # elif [ ${path_itr} = /home/data_storage/mimic-cxr/models/pre-train/medvill_seed2468/36/pytorch_model.bin ]; then
    #     echo "${path_itr} this will be trained!"
    #     python $(dirname "$0")/finetune.py --model_recover_path ${path_itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad chest
    #     # python $(dirname "$0")/finetune.py --model_recover_path ${path_itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad abd
    #     # python $(dirname "$0")/finetune.py --model_recover_path ${path_itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad head
    #     python $(dirname "$0")/finetune.py --model_recover_path ${path_itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad all
    # elif [ ${path_itr} = /home/data_storage/mimic-cxr/models/pre-train/medvill_seed1369/49/pytorch_model.bin ]; then
    #     echo "${path_itr} this will be trained!"
    #     python $(dirname "$0")/finetune.py --model_recover_path ${path_itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad chest
    #     # python $(dirname "$0")/finetune.py --model_recover_path ${path_itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad abd
    #     # python $(dirname "$0")/finetune.py --model_recover_path ${path_itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad head
    #     python $(dirname "$0")/finetune.py --model_recover_path ${path_itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad all
    # elif [ ${path_itr} = /home/data_storage/mimic-cxr/models/pre-train/medvill_seed1234/49/pytorch_model.bin ]; then
    #     echo "${path_itr} this will be trained!"
    #     python $(dirname "$0")/finetune.py --model_recover_path ${path_itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad chest
    #     # python $(dirname "$0")/finetune.py --model_recover_path ${path_itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad abd
    #     # python $(dirname "$0")/finetune.py --model_recover_path ${path_itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad head
    #     python $(dirname "$0")/finetune.py --model_recover_path ${path_itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad all
    # elif [ ${path_itr} = /home/data_storage/mimic-cxr/models/pre-train/medvill_seed1004/49/pytorch_model.bin ]; then
    #     echo "${path_itr} this will be trained!"
    #     python $(dirname "$0")/finetune.py --model_recover_path ${path_itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad chest
    #     # python $(dirname "$0")/finetune.py --model_recover_path ${path_itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad abd
    #     # python $(dirname "$0")/finetune.py --model_recover_path ${path_itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad head
    #     python $(dirname "$0")/finetune.py --model_recover_path ${path_itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad all
    # elif [ ${path_itr} = /home/data_storage/mimic-cxr/models/pre-train/medvill_full_img/pytorch_model.bin ]; then
    #     echo "${path_itr} this will be trained!"
    #     python $(dirname "$0")/finetune.py --model_recover_path ${path_itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad chest
    #     # python $(dirname "$0")/finetune.py --model_recover_path ${path_itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad abd
    #     # python $(dirname "$0")/finetune.py --model_recover_path ${path_itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad head
    #     python $(dirname "$0")/finetune.py --model_recover_path ${path_itr} --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad all
    # else
    #     echo "${itr} this will be passed!"
    # fi
    done


#Command:
#CUDA_VISIBLE_DEVICES=5,6 sh downstream_task/report_generation_and_vqa/vqa_train.sh /home/edlab/jhmoon/mimic_mv_real/mimic-cxr/pre-train