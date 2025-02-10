#!/bin/bash

# Define dataset arguments
DATASET='mixed_rhm'
NUM_FEATURES=16
NUM_CLASSES=16
FRACTION_RULES=0.25
RULE_SEQUENCE_TYPE=1
NUM_LAYERS=2
SEED_RULES=2362346
TRAIN_SIZE=8064
BATCH_SIZE=128
TEST_SIZE=8320
SEED_SAMPLE=34534
INPUT_FORMAT="onehot"
WHITENING=1

# Define architecture arguments
MODEL="hcnn_mixed"
DEPTH=2
WIDTH=256
#BIAS_FLAG="--bias"  # Use "--bias" to enable, remove it if not needed
SEED_MODEL=359

# Define training arguments
LEARNING_RATE=1.0
OPTIMIZER="sgd"
ACCUMULATION=1
MOMENTUM=0.0
#SCHEDULER=None
SCHEDULER_TIME=1024
MAX_EPOCHS=1024

# Define output arguments
PRINT_FREQ=64
SAVE_FREQ=3
#CHECKPOINTS_FLAG="--checkpoints"  # Use "--checkpoints" to enable, remove it if not needed
LOSS_THRESHOLD=0.001
OUTNAME="results.pkl"

# Run the experiment with all arguments
grun python main_mixed.py \
    --device cuda \
    --dataset "$DATASET" \
    --num_features "$NUM_FEATURES" \
    --num_classes "$NUM_CLASSES" \
    --fraction_rules "$FRACTION_RULES" \
    --rule_sequence_type "$RULE_SEQUENCE_TYPE" \
    --num_layers "$NUM_LAYERS" \
    --seed_rules "$SEED_RULES" \
    --train_size "$TRAIN_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --test_size "$TEST_SIZE" \
    --seed_sample "$SEED_SAMPLE" \
    --input_format "$INPUT_FORMAT" \
    --whitening "$WHITENING" \
    --model "$MODEL" \
    --depth "$DEPTH" \
    --width "$WIDTH" \
    $BIAS_FLAG \
    --seed_model "$SEED_MODEL" \
    --lr "$LEARNING_RATE" \
    --optim "$OPTIMIZER" \
    --accumulation "$ACCUMULATION" \
    --momentum "$MOMENTUM" \
    --scheduler_time "$SCHEDULER_TIME" \
    --max_epochs "$MAX_EPOCHS" \
    --print_freq "$PRINT_FREQ" \
    --save_freq "$SAVE_FREQ" \
    --loss_threshold "$LOSS_THRESHOLD" \
    --outname "$OUTNAME"
