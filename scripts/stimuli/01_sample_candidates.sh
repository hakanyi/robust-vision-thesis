#!/bin/bash
#SBATCH --job-name=sample-mooneys
#SBATCH --output=slurm.out
#SBATCH --mail-type=ALL
#SBATCH -G 1
#SBATCH --time=00:40:00        # Time limit hrs:min:sec
#SBATCH --array=0-31

pwd; hostname; date

./run.sh python scripts/stimuli/sample_images.py \
    -e experiments/change-detection/mooney-bodies/ -sd sample-lights-3 \
    --scene_id $SLURM_ARRAY_TASK_ID -n 500
