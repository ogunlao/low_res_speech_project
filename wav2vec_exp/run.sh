#!/bin/bash

pip3 install soundfile
git clone https://github.com/pytorch/fairseq
cd fairseq
pip3 install --editable ./


  
# fairseq-hydra-train \
#     task.data=/content/weak_supervision/data \
#     distributed_training.distributed_world_size=1 \
#     +optimization.update_freq='[128]' \
#     optimization.max_epoch=1 \
#     checkpoint.restore_file=/content/weak_supervision/low_res_speech_project/wav2vec_exp/outputs/2021-04-07/10-56-16/checkpoints/checkpoint_best.pt \
#     --config-dir ./fairseq/examples/wav2vec/config/pretraining \
#     --config-name wav2vec2_large_librivox \

    #--checkpoint-save-dir=/content/weak_supervision/models \
    #--max-epoch 1 \
    #--max-update 1 \
    #--wer_char_level True

# fairseq-hydra-train \
#     task.data=/content/weak_supervision/data \
#     distributed_training.distributed_world_size=1 \
#     --config-dir /content/weak_supervision/low_res_speech_project/wav2vec_exp/fairseq/examples/wav2vec/config/pretraining \
#     --config-name wav2vec2_large_librivox

## Run this to finetune

fairseq-hydra-train \
    task.data=/content/weak_supervision/data/ \
    distributed_training.distributed_world_size=1 \
    optimization.update_freq='[128]' \
    optimization.max_epoch=2 \
    model.w2v_path=/content/weak_supervision/models/checkpoint_best.pt \
    task.normalize=True \
    --config-dir /content/weak_supervision/low_res_speech_project/wav2vec_exp/fairseq/examples/wav2vec/config/finetuning \
    --config-name base_100h \

fairseq-hydra-train \
    task.data=/content/weak_supervision/data/ \
    distributed_training.distributed_world_size=1 \
    optimization.update_freq='[128]' \
    optimization.max_epoch=2 \
    model.w2v_path=/content/weak_supervision/models/checkpoint_best.pt \
    task.normalize=True \
    checkpoint.best_checkpoint_metric="loss" \
    --config-dir /content/weak_supervision/low_res_speech_project/wav2vec_exp/fairseq/examples/wav2vec/config/finetuning \
    --config-name base_100h 