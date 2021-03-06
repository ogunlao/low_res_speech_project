import torch
from argparse import Namespace

args = Namespace(
    
    SEED = 1234,
    
    # dataloader parameters
    DATA_FOLDER = 'data1',
    SAMPLED_DATA_FOLDER = 'clips_16k', 
    
    TRAIN_PS_CSV = 'train_ps.csv',
    VAL_PS_CSV = 'val_ps.csv',
    
    FINETUNE_CSV = 'finetune.csv',
    VAL_CSV = 'val_resampled.csv',
    TEST_CSV = 'test_resampled.csv',
    
    FINETUNE_TSV = 'finetune.tsv',
    VAL_TSV = 'valid.tsv',
    TEST_TSV = 'test.tsv',
    
    RAW_AUDIO_PATH = 'cv-corpus-6.1-2020-12-11/rw/clips/',
    DURATION_SAV_FILE = 'clips_duration.txt',

    TRAIN_DURATION = 100*3600, # hrs
    FINETUNE_DURATION = 50*3600, # hrs
    VALIDATION_DURATION = 20*3600, # hrs

    MODEL_NAME = 'Jasper10x5Dr-En',


    CONFIG_PATH = 'configs/jasper_10x5dr.yaml',

    SAV_CHECKPOINT_PATH = 'models',
    NUMB_GPU = 0,
    MAX_EPOCHS = 15,
)

args.NUMB_GPU = torch.cuda.device_count()
