from argparse import Namespace
import torch

args = Namespace(
    # parameters for sampling
    # SHUFFLE_SAMPLES=True,
    
    SEED = 1234,
    # FREEZE_ENCODER=False,
    # DROPOUT=0.2,
    # WARMUP_PERIOD=1000,
    
    # VAL_DF = None,
    # PRINT_SAMPLE_PS = True,
    USE_PRETRAINED = True,
    EVALUATION_BS = 16, 


    

    # dataloader parameters
    DATA_FOLDER = 'data1', 
    SAMPLED_DATA_FOLDER = 'clips_16k', 
    RAW_AUDIO_PATH = 'cv-corpus-6.1-2020-12-11/rw/clips/',
    DURATION_SAV_FILE = 'clips_duration.txt', 
    MODEL_NAME = 'Jasper10x5Dr-En',

    TRAIN_DURATION = 100 * 3600, # hrs
    FINETUNE_DURATION = 50 * 3600, # hrs
    VALIDATION_DURATION = 20 * 3600, # hrs
  
    TRAIN_PS_CSV = 'train_ps.csv',
    VAL_PS_CSV = 'val_ps.csv',
    FINETUNE_CSV = 'finetune.csv',
    
    # File paths to pseudolabel experiments
    TRAIN_W_PS_CSV = 'train_w_ps.csv',
    VAL_W_PS_CSV = 'val_w_ps.csv',

    #File paths to phonemes
    TRAIN_W_Ph_CSV = 'train_w_ph.csv',
    VAL_W_Ph_CSV = 'val_w_ph.csv',

    ############################################### 2nd experiments with 250train, 150, 50 finetune, 20hrs val#############################
    # TRAIN_DURATION = 250 * 3600, # hrs
    # FINETUNE_DURATION = 150 * 3600, # hrs
    # VALIDATION_DURATION = 20 * 3600, # hrs

    # TRAIN_PS_CSV = 'train_ps_250hrs.csv',
    # VAL_PS_CSV = 'val_ps_20hrs_2nd.csv',
    # FINETUNE_CSV = 'finetune_150hrs.csv',

    # # File paths to pseudolabel experiments
    # TRAIN_W_PS_CSV = 'train_w_ps_250hrs.csv',
    # VAL_W_PS_CSV = 'val_w_ps_20hrs.csv',

    # # File paths to phonemes experiments
    # TRAIN_W_Ph_CSV = 'train_w_ph_250hrs.csv',
    # VAL_W_Ph_CSV = 'val_w_ph_20hrs_2nd.csv',

    # File paths to finetune experiments
    TRAIN_FT_CSV = 'finetune.csv',
    VAL_FT_CSV = 'dev.tsv',
    TEST_FT_CSV = 'test.tsv',

    CONFIG_PATH = 'configs/jasper_10x5dr.yaml',
    CONFIG_PATH_FT = 'configs/jasper_10x5dr_ft.yaml',

    SAV_CHECKPOINT_PATH = 'models',
    NUMB_GPU = 4, # Default number of gpus
    MAX_EPOCHS = 300,
    PATIENCE = 20,

    FREEZE_FEATURE_EXTRACTOR = False,
)
# args_ctc.DEVICE = True if torch.cuda.is_available() else False
# args_ctc.CUT_OFF_TOP_N = len(args_ctc.CHARS)

args.NUMB_GPU = torch.cuda.device_count()
