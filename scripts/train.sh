#/!bin/bash

export PYTHONPATH=./:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1

model=$1
num_gpus=$2

# num_gpus=2
use_port=2681
# model=resnet34
# model=resnet152
# model=vgg19
# model=densenet121
# model=densenet161
# model=resnet50_inat
train_batch_size=80
test_batch_size=150

seed=1028
opt=adam
lr=1e-4

warmup_lr=1e-4
warmup_epochs=5

decay_epochs=3
decay_rate=0.2
sched=step
epochs=12
output_dir=output_cosine/
input_size=224
dim=64
# Loss
features_lr=$lr
add_on_layers_lr=3e-3
prototype_vectors_lr=3e-3

use_ortho_loss=True
ortho_coe=1e-4
consis_coe=0.50
consis_thresh=0.10

ft=train

for data_set in CUB2011;
do
    prototype_num=2000
    data_path=datasets/cub200_cropped
    
    python -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port=$use_port --use_env main.py \
        --seed=$seed \
        --output_dir=$output_dir/$data_set/$model/$seed-$lr-$opt-$epochs-$ft \
        --data_set=$data_set \
        --data_path=$data_path \
        --train_batch_size=$train_batch_size \
        --test_batch_size=$test_batch_size \
        --base_architecture=$model \
        --input_size=$input_size \
        --prototype_shape $prototype_num $dim 1 1 \
        --use_ortho_loss=$use_ortho_loss \
        --ortho_coe=$ortho_coe \
        --consis_coe=$consis_coe \
        --consis_thresh=$consis_thresh \
        --opt=$opt \
        --sched=$sched \
        --lr=$lr \
        --features_lr=$features_lr \
        --add_on_layers_lr=$add_on_layers_lr \
        --prototype_vectors_lr=$prototype_vectors_lr \
        --epochs=$epochs \
        --warmup_epochs=$warmup_epochs \
        --decay_epochs=$decay_epochs \
        --decay_rate=$decay_rate
done