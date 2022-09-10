#!/bin/bash
model_path=$1

for itr in $(find ${model_path}/* -name pytorch_model.bin);
do
    echo ""
    echo ${itr}
    if [ ${itr} = /home/data_storage/mimic-cxr/models/pre-train/medvill_resnet101/11/pytorch_model.bin ]; then
        echo "${itr} this will be trained!"
        # python -m torch.distributed.launch --nproc_per_node=1 --master_port 2132 --use_env downstream_task/report_generation_and_vqa/finetune.py --num_train_epochs 50 --train_batch_size 16 --tasks report_generation --generation_dataset mimic-cxr --mask_prob 0.15 --s2s_prob 1 --bi_prob 0 --model_recover_path ${itr} 
        python -m torch.distributed.launch --nproc_per_node=1 --master_port 34221 --use_env downstream_task/report_generation_and_vqa/finetune.py --num_train_epochs 50 --train_batch_size 16 --tasks report_generation --generation_dataset openi --mask_prob 0.15 --s2s_prob 1 --bi_prob 0 --model_recover_path ${itr} 
    # elif [ ${itr} = /home/data_storage/mimic-cxr/models/pre-train/medvill_seed2468/36/pytorch_model.bin ]; then
    #     echo "${itr} this will be trained!"
    #     python downstream_task/report_generation_and_vqa/finetune.py --num_train_epochs 50 --train_batch_size 128 --tasks report_generation --generation_dataset mimic-cxr --mask_prob 0.15 --s2s_prob 1 --bi_prob 0 --model_recover_path ${itr} 
    #     python downstream_task/report_generation_and_vqa/finetune.py --num_train_epochs 50 --train_batch_size 128 --tasks report_generation --generation_dataset openi --mask_prob 0.15 --s2s_prob 1 --bi_prob 0 --model_recover_path ${itr}
    # elif [ ${itr} = /home/data_storage/mimic-cxr/models/pre-train/medvill_seed1369/49/pytorch_model.bin ]; then
    #     echo "${itr} this will be trained!"
    #     python downstream_task/report_generation_and_vqa/finetune.py --num_train_epochs 50 --train_batch_size 128 --tasks report_generation --generation_dataset mimic-cxr --mask_prob 0.15 --s2s_prob 1 --bi_prob 0 --model_recover_path ${itr} 
    #     python downstream_task/report_generation_and_vqa/finetune.py --num_train_epochs 50 --train_batch_size 128 --tasks report_generation --generation_dataset openi --mask_prob 0.15 --s2s_prob 1 --bi_prob 0 --model_recover_path ${itr}
    # elif [ ${itr} = /home/data_storage/mimic-cxr/models/pre-train/medvill_seed1234/49/pytorch_model.bin ]; then
    #     echo "${itr} this will be trained!"
    #     python downstream_task/report_generation_and_vqa/finetune.py --num_train_epochs 50 --train_batch_size 128 --tasks report_generation --generation_dataset mimic-cxr --mask_prob 0.15 --s2s_prob 1 --bi_prob 0 --model_recover_path ${itr} 
    #     python downstream_task/report_generation_and_vqa/finetune.py --num_train_epochs 50 --train_batch_size 128 --tasks report_generation --generation_dataset openi --mask_prob 0.15 --s2s_prob 1 --bi_prob 0 --model_recover_path ${itr}
    # elif [ ${itr} = /home/data_storage/mimic-cxr/models/pre-train/medvill_seed1004/49/pytorch_model.bin ]; then
    #     echo "${itr} this will be trained!"
    #     python downstream_task/report_generation_and_vqa/finetune.py --num_train_epochs 50 --train_batch_size 128 --tasks report_generation --generation_dataset mimic-cxr --mask_prob 0.15 --s2s_prob 1 --bi_prob 0 --model_recover_path ${itr} 
    #     python downstream_task/report_generation_and_vqa/finetune.py --num_train_epochs 50 --train_batch_size 128 --tasks report_generation --generation_dataset openi --mask_prob 0.15 --s2s_prob 1 --bi_prob 0 --model_recover_path ${itr}
    # # elif [ ${itr} = /home/data_storage/mimic-cxr/models/pre-train/medvill_full_img/pytorch_model.bin ]; then
    # #     echo "${itr} this will be trained!"
    # #     python downstream_task/report_generation_and_vqa/finetune.py --num_train_epochs 50 --train_batch_size 128 --tasks report_generation --generation_dataset mimic-cxr --mask_prob 0.15 --s2s_prob 1 --bi_prob 0 --model_recover_path ${itr} 
    # #     python downstream_task/report_generation_and_vqa/finetune.py --num_train_epochs 50 --train_batch_size 128 --tasks report_generation --generation_dataset openi --mask_prob 0.15 --s2s_prob 1 --bi_prob 0 --model_recover_path ${itr}
    else
        echo "${itr} this will be passed!"
    fi
done

#LOAD_MODELS=''

#command:
#sh downstream_task/report_generation_and_vqa/run.sh /home/edlab/jhmoon/mimic_mv_real/mimic-cxr/pre-train/further