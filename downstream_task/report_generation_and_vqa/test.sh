# #!/bin/bash
model_path=$1
export PYTHONPATH=/home/jhmoon/MedViLL/downstream_task/report_generation_and_vqa/chexpert_labeler/NegBio:$PYTHONPATH
# for itr in $(find ${model_path}/* -name model.50.bin);
# for itr in $(find ${model_path}/* -name pytorch_model.bin);
# for itr in $(find ${model_path}/* -name "*.bin" -print0 |xargs -r -0 ls -1 -t | head -1);
for itr in $(find ${model_path}/* -name "model.50.bin");# -print0 |xargs -r -0 ls -1 -t | head -1);
do
    echo ""
    echo ${itr}
    python $(dirname "$0")/generation_decode.py --model_recover_path ${itr} --beam_size 1
done
        # --num-gpus 2 MODEL.WEIGHTS "/home/jhmoon/git/Optimal-transport/moco_align_uniform/resnet18_dense_stl10_results/exp1/768b_0.36lr_0.1aw_2a_0.8uw_2t_1nw_0.1t/detect.pkl" OUTPUT_DIR "/home/jhmoon/git/Optimal-transport/moco_align_uniform/resnet18_dense_stl10_results/exp1/768b_0.36lr_0.1aw_2a_0.8uw_2t_1nw_0.1t/det"

# /home/edlab/jhmoon/mimic_mv_real/mimic-cxr/downstream_model/report_generation/base_mimic_par_256_128
# /home/edlab/jhmoon/mimic_mv_real/mimic-cxr/downstream_model/report_generation/base_bi_mimic
# /home/edlab/jhmoon/mimic_mv_real/mimic-cxr/downstream_model/report_generation/base_noncross_openi
# /home/edlab/jhmoon/mimic_mv_real/mimic-cxr/downstream_model/report_generation/base_s2s_openi
# /home/edlab/jhmoon/mimic_mv_real/mimic-cxr/downstream_model/report_generation/base_vlp_openi
