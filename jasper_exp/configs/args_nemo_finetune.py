from argparse import Namespace
import torch

args = Namespace(
    EXPERIMENT_TAG='38',
    LANGUAGE='kb',
    SEED = 1234,

    USE_PRETRAINED = True,
    EVALUATION_BS = 16, 

    # dataloader parameters
    DATA_FOLDER = 'data', 
    SAMPLED_DATA_FOLDER = 'clips_16k', 
    RAW_AUDIO_PATH = 'cv-corpus-6.1-2020-12-11/rw/clips/',
    DURATION_SAV_FILE = 'clips_duration.txt', 
    
    MODEL_NAME = 'Jasper10x5Dr-En',    
    
    TRAIN_MANIFEST = 'train_manifest_kb_10hrs_ft.json',
    VAL_MANIFEST = 'val_manifest_kb_ft.json',
    TEST_MANIFEST = 'test_manifest_kb_ft.json',
    
    MONITOR = 'validation_wer',
    DISTRIBUTED_BACKEND = 'ddp',
    RESUME_CHECKPOINT_NAME = None, # full model path

    CONFIG_PATH = 'configs/jasper_10x5dr_kb_ft.yaml',

    SAV_CHECKPOINT_PATH = 'models_kb',
    
    NUMB_GPU = 1, # Default number of gpus
    MAX_EPOCHS = 300,
    PATIENCE = 20,

    FREEZE_FEATURE_EXTRACTOR = False,
)

args.NUMB_GPU = torch.cuda.device_count()

