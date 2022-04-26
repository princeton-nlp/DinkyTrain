#!/bin/bash

####### Usage #######
# Environment variables required
#     DATA_DIR: path to the data directory
#     TASK: glue task name (among mnli qnli qqp rte sst2 mrpc cola stsb) 
#     LR: learning rate     
#     BSZ: batch size
#     EPOCHS: number of epochs
#     SEED: random seed     
#     CKPT_DIR: checkpoint's directory
#     CKPT_NAME: checkpoint's name
#     (Optional) DEEPSPEED=1: use DeepSpeed
# 
####### Usage #######

ARCH=roberta_large
BATCH_SIZE_PER_DEVICE=4  # For RTX 2080. Change based on device.

echo "*** Ckpt: ${CKPT_DIR}/${CKPT_NAME} ***"
echo "*** Task: ${TASK} ***"
echo "*** Learning rate: ${LR}, Batch size: ${BSZ}, #Epochs: ${EPOCHS} ***"

EXTRA_ARGS=""
TASK_ARGS=""
VALID_ARGS="--valid-subset valid"

if [ "$TASK" = "mnli" ]
then
    NUM_CLASSES=3
    TRAIN_EXAMPLES=392702
    VALID_ARGS="--valid-subset valid,valid-mm"
elif [ "$TASK" = "qnli" ]
then
    NUM_CLASSES=2
    TRAIN_EXAMPLES=104743
elif [ "$TASK" = "qqp" ]
then
    NUM_CLASSES=2
    TRAIN_EXAMPLES=363846
elif [ "$TASK" = "rte" ]
then
    NUM_CLASSES=2
    TRAIN_EXAMPLES=2490
elif [ "$TASK" = "sst2" ]
then
    NUM_CLASSES=2
    TRAIN_EXAMPLES=67349
elif [ "$TASK" = "mr" ]
then
    NUM_CLASSES=2
    TRAIN_EXAMPLES=8662
elif [ "$TASK" = "mrpc" ]
then
    NUM_CLASSES=2
    TRAIN_EXAMPLES=3668
elif [ "$TASK" = "cola" ]
then
    NUM_CLASSES=2
    TRAIN_EXAMPLES=8551
elif [ "$TASK" = "stsb" ]
then
    NUM_CLASSES=1
    TRAIN_EXAMPLES=5749
    TASK_ARGS="--regression-target"
fi

if [ "$DEEPSPEED" = "1" ]
then
    ARCH="deepspeed_$ARCH"
fi

UPDATE_FREQ=$(expr ${BSZ} / $BATCH_SIZE_PER_DEVICE)
TOTAL_UPDATES=$(expr ${EPOCHS} \* $TRAIN_EXAMPLES / ${BSZ})

DATA_DIR="${DATA_DIR}/${TASK%-*}/bin"
FINETUNE_DIR="${CKPT_DIR}/ft-${CKPT_NAME}-${TASK}"

python fairseq_cli/train.py $DATA_DIR \
    --finetune-from-model $CKPT_DIR/$CKPT_NAME \
    --save-dir $FINETUNE_DIR \
    --seed $SEED \
    \
    --arch $ARCH \
    --task sentence_prediction \
    --criterion sentence_prediction \
    \
    --max-positions 512 \
    --batch-size $BATCH_SIZE_PER_DEVICE \
    --required-batch-size-multiple 1 \
    --update-freq $UPDATE_FREQ \
    --num-classes $NUM_CLASSES \
    --total-num-update $TOTAL_UPDATES \
    --max-epoch $EPOCHS \
    \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.1 \
    --optimizer adam \
    --adam-betas "(0.9, 0.98)" \
    --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay \
    --lr $LR \
    --warmup-updates $(expr $TOTAL_UPDATES \* 6 / 100) \
    --find-unused-parameters \
    --keep-last-epochs 1 \
    \
    --encoder-normalize-before \
    \
    --no-epoch-checkpoints \
    --log-format simple \
    --log-interval 10 \
    \
    $EXTRA_ARGS \
    $TASK_ARGS \
    $VALID_ARGS 


EVAL_CKPT=$FINETUNE_DIR/checkpoint_last.pt

python fairseq_cli/validate_glue.py $DATA_DIR \
    --path $EVAL_CKPT \
    --task sentence_prediction \
    --criterion sentence_prediction \
    --batch-size $BATCH_SIZE_PER_DEVICE \
    --required-batch-size-multiple 1 \
    --num-classes $NUM_CLASSES \
    --results-path $FINETUNE_DIR/${TASK}.json \
    $VALID_ARGS

