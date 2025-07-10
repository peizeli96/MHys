# The name of experiment
name=VLBart

output=snap/vqa/$name

PYTHONPATH=$PYTHONPATH:./src \
python src/vqa.py \
        --train retvqa_release_v1_trainval_vlt5 \
        --valid retvqa_release_v1_test_vlt5 \
        --test retvqa_release_v1_test_vlt5 \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 1e-4 \
        --epochs 20 \
        --num_workers 4 \
        --backbone bart_path/ \
        --individual_vis_layer_norm False \
        --output $output ${@:2} \
        --load snap/pretrain/VLBart/Epoch30 \
        --num_beams 5 \
        --batch_size 24 \
        --valid_batch_size 8 \
