OUTDIR="/remote-home/share/data200/172.16.11.200/zhengqiaoyu/RadNet_KE/Logout/logTest"
PORT=29500

START=0
END=5569
BACKBONE="resnet"
LEVEL="articles"
DEPTH=32
LTYPE="MultiLabel"
AUGMENT=True
N_image=6
FUSE="late"
KE=True
ADAPTER=False
CHECKPOINT="None"
SAFETENSOR="/remote-home/share/data200/172.16.11.200/zhengqiaoyu/MedVision/Logout/log10_0_5569_res_art_32_BCE_True_comb_10/output/checkpoint-46650/pytorch_model.bin"

WORKER=24
BATCHSIZE=2
LR=8e-6

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
nohup torchrun --nproc_per_node=6 --master_port $PORT /remote-home/share/data200/172.16.11.200/zhengqiaoyu/RadNet_KE/eval.py \
    --output_dir $OUTDIR/output \
    --num_train_epochs 100 \
    --per_device_train_batch_size $BATCHSIZE \
    --per_device_eval_batch_size $BATCHSIZE \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --learning_rate $LR \
    --save_total_limit 4 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --run_name RadNet \
    --ignore_data_skip true \
    --dataloader_num_workers $WORKER \
    --remove_unused_columns False \
    --metric_for_best_model "eval_loss" \
    --load_best_model_at_end True \
    --report_to "none" \
    --start_class $START \
    --end_class $END \
    --backbone $BACKBONE \
    --level $LEVEL \
    --depth $DEPTH \
    --ltype $LTYPE \
    --augment $AUGMENT \
    --n_image $N_image \
    --fuse $FUSE \
    --ke $KE \
    --adapter $ADAPTER \
    --checkpoint $CHECKPOINT \
    --safetensor $SAFETENSOR \
>> $OUTDIR/output.log 2>&1 &