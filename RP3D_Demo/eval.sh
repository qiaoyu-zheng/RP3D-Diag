OUTDIR=".../RP3D_Demo/Logout/log0Eval"
PORT=29300

START=0
END=5569
BACKBONE="resnet"
LEVEL="articles"
SIZE=256
DEPTH=32
LTYPE="MultiLabel"
AUGMENT=True
N_image=4
N_aug=2
Prob=0.7
Dim=0
Hid_Dim=2048
FUSE="early"
MIX=False
KE=False
ADAPTER=False
CHECKPOINT="None"
SAFETENSOR=".../RP3D_Demo/Logout/RP3D_5569_res_ART_256_32_BCE_T_4_2_0.7_2048_E_F_F_F_N/output/pytorch_model.bin"
# SAFETENSOR="None"

WORKER=1
BATCHSIZE=1
LR=1e-5

export CUDA_VISIBLE_DEVICES=0
# export http_proxy=http://172.16.6.115:18080  
# export https_proxy=http://172.16.6.115:18080 
nohup torchrun --nproc_per_node=1 --master_port $PORT .../RP3D_Demo/eval.py \
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
    --size $SIZE \
    --depth $DEPTH \
    --ltype $LTYPE \
    --augment $AUGMENT \
    --n_image $N_image \
    --n_aug $N_aug \
    --prob $Prob \
    --dim $Dim \
    --hid_dim $Hid_Dim \
    --fuse $FUSE \
    --mix $MIX \
    --ke $KE \
    --adapter $ADAPTER \
    --checkpoint $CHECKPOINT \
    --safetensor $SAFETENSOR \
>> $OUTDIR/output.log 2>&1 &