#!/bin/bash

#SBATCH --job-name mixed_rhm_v_16_L_2_P_8064_lr_1_randn_a   # Assign unique name
#SBATCH --chdir /scratch/parley/   # go to scratch
#SBATCH -o /home/parley/running/%A.%x_%a.out # STDOUT
#SBATCH -e /home/parley/running/%A.%x_%a.err # STDERR

#SBATCH --partition h100
#SBATCH --time 00:10:00
#SBATCH --mem 90G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres gpu:1
#SBATCH --account pcsl
#SBATCH --array=0-7%8


# Define dataset arguments

seed1=$(od -An -N3 -i /dev/random)
#seed1=100
#seed2=100
seed2=$(od -An -N3 -i /dev/random)
seed3=$(od -An -N3 -i /dev/random)
#seed3=100


DATASET='mixed_rhm'
NUM_FEATURES=16
NUM_CLASSES=16
FRACTION_RULES=0.25
RULE_SEQUENCE_TYPE=1
NUM_LAYERS=2
NUM_TOKENS=5
#SEED_RULES=seed1
TRAIN_SIZE=8064
BATCH_SIZE=128
#TEST_SIZE=768
MAX_DATA=16384
#SEED_SAMPLE=seed2
INPUT_FORMAT="onehot"
WHITENING=1

# Define architecture arguments
MODEL="hcnn_mixed"
DEPTH=2
WIDTH=256
#BIAS_FLAG="--bias"  # Use "--bias" to enable, remove it if not needed
#SEED_MODEL=seed3

# Define training arguments
LEARNING_RATE=1.0
OPTIMIZER="sgd"
ACCUMULATION=1
MOMENTUM=0.0
#SCHEDULER=None
SCHEDULER_TIME=1024
MAX_EPOCHS=2048

# Define output arguments
PRINT_FREQ=64
SAVE_FREQ=3
CHECKPOINTS_FLAG="--checkpoints"  # Use "--checkpoints" to enable, remove it if not needed
LOSS_THRESHOLD=0.001
OUTNAME="dynamics_run_6_122_${SLURM_ARRAY_TASK_ID}.pkl"


path="${SLURM_JOB_NAME}" # create directory with ID
mkdir -p $path
cd $path

echo STARTING AT
date


# Run the experiment with all arguments
srun python /home/parley/Mixed_RHM/main_mixed.py \
    --device cuda \
    --dataset "$DATASET" \
    --num_features "$NUM_FEATURES" \
    --num_classes "$NUM_CLASSES" \
    --num_tokens "$NUM_TOKENS" \
    --fraction_rules "$FRACTION_RULES" \
    --rule_sequence_type "$RULE_SEQUENCE_TYPE" \
    --num_layers "$NUM_LAYERS" \
    --seed_rules ${seed1} \
    --train_size "$TRAIN_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --max_data "$MAX_DATA" \
    --seed_sample ${seed2} \
    --input_format "$INPUT_FORMAT" \
    --whitening "$WHITENING" \
    --model "$MODEL" \
    --depth "$DEPTH" \
    --width "$WIDTH" \
    $BIAS_FLAG \
    $CHECKPOINTS_FLAG \
    --seed_model ${seed3} \
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



echo FINISHED AT
date