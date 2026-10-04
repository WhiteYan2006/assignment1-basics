#!/bin/bash
# TinyStories 实验脚本：只定义 run_exp 一个函数，跑哪些实验由底部的命令决定
# 用法（Linux 下可以直接执行；windows 下要先安装 git，在自带的 gitbash 终端里执行）:
#   1. 把底部「实验命令」里要跑的行取消注释
#   2. nohup bash scripts/run_experiments.sh > logs/ablation.txt 2>&1 &
#   也可以 source 后手动调用:
#   source scripts/run_experiments.sh
#   run_exp "lr_sweep" 3e-4 3e-5 64
#   WANDB_MODE=offline bash scripts/run_experiments.sh   # 不上传云端，只写本地 ./wandb

export CUDA_VISIBLE_DEVICES=0
source "/e/Anaconda/etc/profile.d/conda.sh"
conda activate base
export PYTHON=python
# conda activate base 执行 Conda 的 Bash 激活脚本。它设置 SSL_CERT_FILE=E:/Anaconda/ssl/cacert.pem，但这个文件不存在。
# W&B 建立 HTTPS 连接时，需要用受信任的 CA 证书验证服务器身份。错误的证书配置导致验证失败，于是报x509: certificate signed by unknown authority
# Conda 的 CMD 激活脚本设置的是正确路径E:/Anaconda/Library/ssl/cacert.pe
export SSL_CERT_FILE="E:/Anaconda/Library/ssl/cacert.pem"

# ========== 公共超参 ==========
train_data=data/TinyStories/TinyStoriesV2-GPT4-train.npy
val_data=data/TinyStories/TinyStoriesV2-GPT4-valid.npy
dataset_name=TinyStoriesV2-GPT4
vocab_size=10000
context_length=128

d_model=512
num_layers=4
num_heads=16
rope_theta=10000

weight_decay=0.1
beta1=0.9
beta2=0.95
max_l2_norm=1.0
seed=2026

warmup_iters=500
total_iters=5000
val_interval=100
save_interval=500
save_dir=results/checkpoints

embed_init_std=0.02

# ========== 核心训练函数 ==========
# run_exp <exp_name> <max_lr> <min_lr> <batch_size> [额外参数...]
# 额外参数可传: --total_iters 40000 --warmup_iters 4000 --cosine_cycle_iters 40000
# 实验名统一拼成 {exp_name}-lr{max_lr}_{min_lr}-bs{bs}-s{seed}，由 --run_name 传给训练脚本，
# 训练脚本据此建目录 checkpoints/{dataset_name}/{run_name}/，wandb 上也用同一个名字
run_exp() {
    local exp_name=$1; shift
    local max_lr=$1; shift
    local min_lr=$1; shift
    local bs=$1; shift
    local extra_args="$@"

    local run_name="${exp_name}-lr${max_lr}_${min_lr}-bs${bs}-s${seed}"
    echo "=== ${run_name} ==="

    $PYTHON train_llm.py \
        --run_name ${run_name} \
        --dataset_name ${dataset_name} \
        --vocab_size ${vocab_size} \
        --context_length ${context_length} \
        --d_model ${d_model} \
        --num_layers ${num_layers} \
        --num_heads ${num_heads} \
        --rope_theta ${rope_theta} \
        --weight_decay ${weight_decay} \
        --beta1 ${beta1} \
        --beta2 ${beta2} \
        --max_l2_norm ${max_l2_norm} \
        --seed ${seed} \
        --warmup_iters ${warmup_iters} \
        --cosine_cycle_iters ${total_iters} \
        --total_iters ${total_iters} \
        --max_lr ${max_lr} \
        --min_lr ${min_lr} \
        --batch_size ${bs} \
        --train_data_path ${train_data} \
        --val_data_path ${val_data} \
        --save_dir ${save_dir} \
        --val_interval ${val_interval} \
        --save_interval ${save_interval} \
        --amp --embed_init_std ${embed_init_std} \
        ${extra_args}
}

# ========== 实验命令 ==========
# 下面全是注释，直接 bash 本文件不会跑任何实验：把要跑的行取消注释即可。

# (0) 试验能否跑起来
run_exp "try_in_bash"  3e-4  3e-5  128

# (1) lr 扫描（bs=128）
# run_exp "lr_sweep"  3e-4  3e-5  128
# run_exp "lr_sweep"  6e-4  6e-5  128
# run_exp "lr_sweep"  3e-3  3e-4  128

# (2) bs 扫描（lr=1e-2→1e-3，固定 token 数 ~82M：bs × 128 × iters 保持一致）
# run_exp "bs_sweep"  1e-2 1e-3   8 --total_iters 5000  --warmup_iters 500  --cosine_cycle_iters 5000
# run_exp "bs_sweep"  1e-2 1e-3   16 --total_iters 40000 --warmup_iters 4000 --cosine_cycle_iters 40000
# run_exp "bs_sweep"  1e-2 1e-3  32 --total_iters 20000 --warmup_iters 2000 --cosine_cycle_iters 20000
# run_exp "bs_sweep"  1e-2 1e-3  64 --total_iters 10000 --warmup_iters 1000 --cosine_cycle_iters 10000
# run_exp "bs_sweep"  1e-2 1e-3  128 --total_iters 5000

# (3) 消融实验（lr=1e-2→1e-3, bs=128，只改结构开关）
# run_exp "ablation_no_rmsnorm"  1e-2 1e-3 128 --no_rmsnorm   --ablation_name no_rmsnorm
# run_exp "ablation_post_norm"   1e-2 1e-3 128 --post_norm    --ablation_name post_norm
# run_exp "ablation_nope"        1e-2 1e-3 128 --rope_theta 0 --ablation_name nope
# run_exp "ablation_no_gated"    1e-2 1e-3 128 --no_gated_ffn --ablation_name no_gated
# run_exp "ablation_tied"        1e-2 1e-3 128 --tie_embedding --ablation_name tied
