#!/bin/bash
#SBATCH --job-name=SVILA-4b
#SBATCH --output=job_outputs/SVILA-4b.out
#SBATCH --error=job_outputs/SVILA-4b.err
#SBATCH --time=24:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal

# --- ENVIRONMENT SETUP ---
module purge
module load profile/base
module load gcc/12.2.0
module load python/3.11.7
module load cuda/12.2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# --- ENVIRONMENT VARIABLES FOR NCCL and PYTHON ---
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=34441
export NCCL_SOCKET_IFNAME=ib0
export TORCH_CUDA_ARCH_LIST="8.0"
export NCCL_PROTO=simple
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000



# --- VIRTUAL ENVIRONMENT ---
. /leonardo_work/ACCOUNT_NAME/venv/bin/activate

# --- SCRIPT ARGUMENTS ---
# You need to load model locally on Leonardo
MODEL="/leonardo_work/ACCOUNT_NAME/models/gemma-3-4b-it"
DATA="sample_data.json"

# --- EXECUTION ---
srun python -u finetune_gemma.py \
    --deepspeed gemma_leonardo_zero2.json \
    --model_id "$MODEL" \
    --data_path "$DATA" \
    --ddp_find_unused_parameters True \
    --remove_unused_columns False \
    --disable_flash_attn2 True \
    --lora_enable False \
    --tune_img_projector True \
    --freeze_vision_tower True \
    --freeze_llm False \
    --output_dir output/SVILA-4b \
    --run_name SVILA-4b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --projector_lr 1e-5 \
    --vision_lr 2e-7 \
    --weight_decay 0.1 \
    --warmup_ratio 0.05 \
    --adam_beta2 0.98 \
    --max_grad_norm 0.5 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --lazy_preprocess False \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --report_to "tensorboard"

deactivate