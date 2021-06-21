#!/bin/bash
#SBATCH -o /home/mila/e/enoch.tetteh/asr/logs/log_%j.out
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64Gb

rsync -avd /home/mila/e/enoch.tetteh/asr/weak_supervision/data $SLURM_TMPDIR

module load anaconda
source activate asr_env
module load cuda/11.0

#RUN your code
python /home/mila/e/enoch.tetteh/asr/weak_supervision/low_res_speech_project/jasper_exp/finetune.py --use-eng-asr
