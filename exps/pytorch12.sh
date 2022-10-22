#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --partition=p4de
#SBATCH --job-name=cu12
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=10
##SBATCH --exclude 'a100-st-p4d24xlarge-459'
#SBATCH --mem=0GB
#SBATCH --signal=USR1@90
#SBATCH --open-mode=append
#SBATCH --time=0
#SBATCH --array=0
#SBATCH --wckey=submitit

# command
#export MODULEPATH=/data/home/vkhalidov/modulefiles:$MODULEPATH
#module load cuda/11.3
#module load nccl/2.12.7-cuda.11.3
#module load nccl_efa/1.2.0-nccl.2.12.7-cuda.11.3
#export SUBMITIT_EXECUTOR=slurm
#source activate mega

source activate mega12
split=1
seeds=(22 42 65537 8191 131071)
seed=${seeds[$split]}

ACTIVATION='silu'
ATTN_ACT='softmax'
LR=5e-3
NORM_TYPE='layernorm'
TRUNCATION=8192
NDIM=16

CHUNK=1024
MIN_MULTIPLE=2
MAX_MULTIPLE=6

WARMUP=24000
WEIGHT_DECAY=0.1
TOTAL_NUM_UPDATES=400000

DATE=`date +%Y%m%d`
SAVE_ROOT=saved_models
DATA=/fsx/chuntinz/data/mega_data/wikitext-103
model=mega_lm_adaptive_big
exp_name=cu11_test
SAVE=${SAVE_ROOT}/${exp_name}
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh

export MASTER_ADDR=${SLURM_NODELIST:0:20}${SLURM_NODELIST:21:1}
export MASTER_PORT=15127
export WORLD_SIZE=16

srun --label python -u train.py ${DATA} \
    --seed ${seed} --ddp-backend no_c10d --max-target-positions 8096 --decoder-hidden-dim 2048 \
    --valid-subset valid --task language_modeling -a ${model} \
    --activation-fn ${ACTIVATION} --attention-activation-fn ${ATTN_ACT} \
    --decoder-n-dim ${NDIM} --decoder-chunk-size ${CHUNK} --normalize-before --no-affine-final-norm \
    --max-tokens 6144 --tokens-per-sample 2048 --update-freq 1 \
    --variant-block-multiple-min ${MIN_MULTIPLE} --variant-block-multiple-max ${MAX_MULTIPLE} \
    --normalization-type ${NORM_TYPE} --truncation-length ${TRUNCATION} --rel-pos-bias "rotary" \
    --optimizer adam --lr ${LR} --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 0.25 \
    --lr-scheduler linear_decay --total-num-update ${TOTAL_NUM_UPDATES} --end-learning-rate 0.0 \
    --warmup-updates ${WARMUP} --warmup-init-lr '1e-07' \
    --criterion adaptive_loss \
    --dropout 0.3 --attention-dropout 0.1 --hidden-dropout 0.1 --weight-decay ${WEIGHT_DECAY} \
    --max-update ${TOTAL_NUM_UPDATES} \
    --no-epoch-checkpoints \
    --sample-break-mode 'complete'\
    --valid-block "splits:10" \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 | tee -a ${SAVE}/log.txt

