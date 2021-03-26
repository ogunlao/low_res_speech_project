import torch
from argparse import Namespace

args = Namespace(
    # parameters for sampling
    # SHUFFLE_SAMPLES=True,
    
    SEED = 1234,
    # FREEZE_ENCODER=False,
    # DROPOUT=0.2,
    # WARMUP_PERIOD=1000,
    
    # VAL_DF = None,
    # PRINT_SAMPLE_PS = True,
    # USE_PRETRAINED = False,
    
    # # parameters for encoder/decoder CTC
    # DIM_ENCODER=256,
    # DIM_CONTEXT=256,
    # KEEP_HIDDEN_VECTOR=False,
    # N_LEVELS_CONTEXT=1,
    # CONTEXT_RNN="LSTM", # "RNN" or "GRU"
    # #N_PREDICTIONS=12,
    # N_NEGATIVE_SAMPLE =128,

    # # data parameters
    # SIZE_WINDOW = 20480,
    # DATA_EXT='.wav',
    # SOURCE_FRAME_RATE=48000,
    # DEST_FRAME_RATE=16000,

    # # training parameters
    # PATIENCE=5, # early stopping limit
    # N_EPOCH=150,
    # CHECKPOINT_PATH = 'checkpoint_data/checkpoint_30.pt',
    # MIN_LR = 0.0,
    # LEARNING_RATE = 1e-3, # set starting lr
    # WEIGHT_DECAY = 0.000,
    # BEAM_WIDTH = 1, # for beam search, set to 1 since we are using max_decoding
    

    # dataloader parameters
    DATA_FOLDER = 'data', 
    RAW_AUDIO_PATH = 'cv-corpus-6.1-2020-12-11/rw/clips/',
    SAMPLED_DATA_FOLDER = 'clips_16k', 
    DURATION_SAV_FILE = 'clips_duration.txt', 

    TRAIN_DURATION = 20, # hrs
    FINETUNE_DURATION = 10, # hrs
    VALIDATION_DURATION = 10, # hrs

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



    # PATH_TRAIN_DATA_CER = "/home/ola/store/clips_16k/",
    # PATH_VAL_DATA_CER = "/home/ola/store/content/val/",
    # PATH_TEST_DATA_CER = "/home/ola/store/content/test",
    # PATH_LETTER_DATA_CER = '/home/ola/store/content/char_to_labels.txt',
    # PATH_PSEUDOLABEL_DATA_CER = '/home/ola/store/content/char_to_labels_ps.txt',

    # TRAIN_BATCH_SIZE=16,
    # VAL_BATCH_SIZE = 8,
    # MAX_TRAINING_DURATION = 2*60*60, # change between 2, 10 and 20 hrs

    # AUDIO_PATH_TRAIN="/home/ola/store/content/train_2h",
    # AUDIO_PATH_DEV = "/home/ola/store/content/val/",
    # AUDIO_PATH_TEST = "/home/ola/store/content/test/",

    # FINAL_MODEL_SAVE_PATH = '/content/drive/My Drive/Colab Notebooks/data/fr/',
    # CHECKPOINT_SAVE_PATH = "/home/ola/store/checkpoint.ckpt",

    # CHARS =["^", "a", "b", "c" , "d" ,"e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", 
    #         "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", " "], #replace with  your alphabet
)

# args_ctc.DEVICE = True if torch.cuda.is_available() else False
# args_ctc.CUT_OFF_TOP_N = len(args_ctc.CHARS)

args.NUMB_GPU = torch.cuda.device_count()
