#!/bin/bash

#SBATCH --job-name mixed_rhm_v_f_L_2_T1_double_sharing_epochs  # Assign unique name
#SBATCH --chdir /scratch/parley/   # go to scratch
#SBATCH -o /home/parley/running_2/%A.%x_%a.out # STDOUT
#SBATCH -e /home/parley/running_2/%A.%x_%a.err # STDERR

#SBATCH --partition h100
#SBATCH --time 4:00:00
#SBATCH --mem 90G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres gpu:1
#SBATCH --account pcsl
#SBATCH --array=0-4%5


# Define dataset arguments

insert_train=$1

seed1=$(od -An -N3 -i /dev/random)
#seed1=100
#seed2=100
seed2=$(od -An -N3 -i /dev/random)
seed3=$(od -An -N3 -i /dev/random)
#seed3=100


DATASET='mixed_rhm'
NUM_FEATURES=$2
NUM_CLASSES=$2
FRACTION_RULES=$3
RULE_SEQUENCE_TYPE=1
NUM_LAYERS=2
NUM_TOKENS=5
#SEED_RULES=seed1
#TRAIN_SIZE=8064
BATCH_SIZE=128
#MAX_DATA=1404928
#TEST_SIZE=768
#SEED_SAMPLE=seed2
INPUT_FORMAT="onehot"
WHITENING=1
NON_OVERLAPPING=0
D_5_SET=1


# Define architecture arguments
MODEL="hcnn_sharing"
DEPTH=2
WIDTH=65536
WIDTH_2=256
#BIAS_FLAG="--bias"  # Use "--bias" to enable, remove it if not needed
#SEED_MODEL=seed3

# Define training arguments
LEARNING_RATE=1.0
OPTIMIZER="sgd"
ACCUMULATION=1
MOMENTUM=0.9
#SCHEDULER=None
SCHEDULER_TIME=1024
MAX_EPOCHS=150
STOPPING_CRITERIA=1
CHECK_PLATEAU=0
LOG_LINEAR_SWITCH=1000000
REPLACEMENT_FLAG="--replacement"

# Define output arguments
PRINT_FREQ=64
SAVE_FREQ=3
CHECKPOINTS_FLAG="--checkpoints"  # Use "--checkpoints" to enable, remove it if not needed
LOSS_THRESHOLD=0.001
OUTNAME="dynamics_v_${NUM_CLASSES}_f_${FRACTION_RULES}_L_2_d_5_set_P_${insert_train}_${SLURM_ARRAY_TASK_ID}.pkl"


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
    --train_size ${insert_train} \
    $REPLACEMENT_FLAG \
    --batch_size "$BATCH_SIZE" \
    --seed_sample ${seed2} \
    --input_format "$INPUT_FORMAT" \
    --whitening "$WHITENING" \
    --non_overlapping "$NON_OVERLAPPING" \
    --d_5_set "$D_5_SET" \
    --model "$MODEL" \
    --depth "$DEPTH" \
    --width "$WIDTH" \
    --width_2 "$WIDTH_2" \
    $BIAS_FLAG \
    --seed_model ${seed3} \
    --lr "$LEARNING_RATE" \
    --optim "$OPTIMIZER" \
    --accumulation "$ACCUMULATION" \
    --momentum "$MOMENTUM" \
    --scheduler_time "$SCHEDULER_TIME" \
    --max_epochs "$MAX_EPOCHS" \
    --stopping_criteria "$STOPPING_CRITERIA" \
    --check_plateau "$CHECK_PLATEAU" \
    --log_linear_switch "$LOG_LINEAR_SWITCH" \
    --print_freq "$PRINT_FREQ" \
    --save_freq "$SAVE_FREQ" \
    --loss_threshold "$LOSS_THRESHOLD" \
    --outname "$OUTNAME"



echo FINISHED AT
date