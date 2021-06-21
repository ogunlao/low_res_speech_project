import torch
from argparse import Namespace

args = Namespace(
    SEED = 1234,    

    # dataloader parameters
    DATA_FOLDER = 'data1', 
    RAW_AUDIO_PATH = 'cv-corpus-6.1-2020-12-11/rw/clips/',
    SAMPLED_DATA_FOLDER = 'clips_16k', 
    DURATION_SAV_FILE = 'clips_duration', 

    TRAIN_DURATION = 100*3600, # secs
    FINETUNE_DURATION = 10*3600, # secs
    VALIDATION_DURATION = 20*3600, # secs

    MODEL_NAME = 'Jasper10x5Dr-En',
    TRAIN_PS_CSV = 'train_ps.csv',
    VAL_PS_CSV = 'val_ps.csv',
    FINETUNE_CSV = 'finetune.csv',

    TRAIN_W_PS_CSV = 'train_w_ps.csv',
    VAL_W_PS_CSV = 'val_w_ps.csv',


    CONFIG_PATH = 'configs/jasper_10x5dr.yaml',

    SAV_CHECKPOINT_PATH = 'models',
    NUMB_GPU = 0,
    MAX_EPOCHS = 15,
    
    SAMPLING_RATE = 16000,
    
)

args.NUMB_GPU = torch.cuda.device_count()
