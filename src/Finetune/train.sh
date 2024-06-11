


OUTDIR=".../src/Finetune/Logout/log1"
PORT=29500
START=0
END=3
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
FUSE="late"
MIX=False
KE=False
ADAPTER=False
CHECKPOINT="None"
SAFETENSOR=".../pytorch_model.bin"
# SAFETENSOR="None"

WORKER=16
BATCHSIZE=16
LR=1e-5

export CUDA_VISIBLE_DEVICES=0
# export NCCL_IGNORE_DISABLED_P2P=1
# export http_proxy=http://172.16.6.115:18080  
# export https_proxy=http://172.16.6.115:18080 
nohup torchrun --nproc_per_node=1 --master_port $PORT .../src/Finetune/train.py \
    --output_dir $OUTDIR/output \
    --num_train_epochs 100 \
    --per_device_train_batch_size $BATCHSIZE \
    --per_device_eval_batch_size $BATCHSIZE \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --learning_rate $LR \
    --save_total_limit 4 \
    --save_safetensors False \
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
    --report_to "wandb" \
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

