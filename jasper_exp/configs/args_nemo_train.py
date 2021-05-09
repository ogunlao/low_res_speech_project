from argparse import Namespace
import torch

args = Namespace(    
    EXPERIMENT_TAG='01',
    LANGUAGE='kb',
    SEED = 1234,
    
    USE_PRETRAINED = True,
    EVALUATION_BS = 16, 

    # dataloader parameters
    DATA_FOLDER = 'data', 
    RAW_AUDIO_PATH = 'cv-corpus-6.1-2020-12-11/rw/clips/',
    SAMPLED_DATA_FOLDER = 'clips_16k', 
    DURATION_SAV_FILE = 'clips_duration.txt', 
    MODEL_NAME = 'Jasper10x5Dr-En',
    MODEL_CHECKPOINT = '',

    CONFIG_PATH = 'configs/jasper_10x5dr.yaml',

    SAV_CHECKPOINT_PATH = 'models',
    NUMB_GPU = 1, # Default number of gpus
    MAX_EPOCHS = 300,
    PATIENCE = 20,

    FREEZE_FEATURE_EXTRACTOR = False,
    
    TRAIN_MANIFEST = 'train_manifest.json',
    VAL_MANIFEST = 'val_manifest.json',
    
    MONITOR = 'val_loss', 
    DISTRIBUTED_BACKEND = 'ddp',
    RESUME_CHECKPOINT_NAME = None, # full model path

)

args.NUMB_GPU = torch.cuda.device_count()

