#!/bin/bash

####### Usage #######
# Environment variables required
#     DATA_DIR: path to the data directory
#     GPU: number of GPUs
#     (Optional) DEEPSPEED=1: use DeepSpeed
# 
# Usage:
#     GPU={number of GPUs} DATA_DIR={data path} [DEEPSPEED=1] bash run_efficient_mlm_recipe.sh
####### Usage #######

####### Hyperparameter ########
BSZ=4096                  # Batch size
PEAK_LR=2e-3              # Learning rate
MASK_RATE=0.4             # Masking rate
SAME_RATE=0               # Replace with same tokens
RANDOM_RATE=0             # Replace with random tokens
TOTAL_UPDATES=23000       # Total number of training steps (~3 epochs with Wikipedia+BookCorpus)
WARMUP_UPDATES=1380       # Warmup the learning rate over this many updates (6%)
TOKENS_PER_SAMPLE=128     # Max sequence length
MAX_SENTENCES=8           # Number of sequences per GPU (for RTX2080, 8 is the max)
ARCH=roberta_large        # Model architecture
JOB_NAME=efficient_roberta_large_bsz4096_lr2e-3_mask0.4 
####### Hyperparameter ########

WANDB_PROJECT=$JOB_NAME                          # Weight and Bias
MAX_POSITIONS=512                                # Num. positional embeddings 
UPDATE_FREQ=$(expr $BSZ / $MAX_SENTENCES / $GPU) # Gradient accumulation
SAVE_INTERVAL_UPDATES=5000                  

EXTRA_ARGS=""
if [ "$DEEPSPEED" = "1" ]
then
    EXTRA_ARGS="--deepspeed-stochastic-mode"     # DeepSpeed stochastic mode (faster). You can turn it off
    ARCH="deepspeed_$ARCH"
fi

fairseq-train  $DATA_DIR \
    --save-dir checkpoints/$JOB_NAME \
    --wandb-project $WANDB_PROJECT  \
    \
    --task masked_lm \
    --criterion masked_lm \
    --arch $ARCH \
    --max-positions $MAX_POSITIONS \
    --encoder-normalize-before \
    \
    --mask-prob $MASK_RATE \
    --leave-unmasked-prob $SAME_RATE \
    --random-token-prob $RANDOM_RATE \
    \
    --sample-break-mode none \
    --tokens-per-sample $TOKENS_PER_SAMPLE \
    \
    --optimizer adam \
    --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay \
    --lr $PEAK_LR \
    --batch-size $MAX_SENTENCES \
    --update-freq $UPDATE_FREQ \
    \
    --warmup-updates $WARMUP_UPDATES \
    --total-num-update $TOTAL_UPDATES \
    --max-update $TOTAL_UPDATES \
    \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    \
    --fp16 \
    --fp16-scale-tolerance 0.1 \
    --fp16-scale-window 50 \
    \
    --keep-interval-updates 100 \
    --save-interval-updates $SAVE_INTERVAL_UPDATES \
    --validate-interval-updates $SAVE_INTERVAL_UPDATES \
    --save-interval 99999999 \
    --validate-interval 99999999 \
    --num-workers 8 \
    --seed 1 \
    --log-format simple \
    --log-interval 1  \
    $EXTRA_ARGS


