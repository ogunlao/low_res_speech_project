import os
from argparse import Namespace
from unicodedata import normalize
from collections import defaultdict
import torch

def make_dirs(path):
    if not os.path.exists(path):
      os.makedirs(path)
      

def calc_distr(df, punct_set):
  chars = defaultdict(int)
  for sentence in df.sentence.values:
    sentence = normalize('NFD', sentence).encode('ascii', 'ignore')
    sentence = sentence.decode('UTF-8')
    for char in sentence:
      if char not in punct_set:
        if char == " ":
          chars["<sep>"]+=1
        else: chars[char.lower()]+=1
  
  #chars_sorted = sorted(chars.items())
  #print(chars_sorted)
  
  return chars

args_preproc = Namespace(
    # parameters for sampling
    SHUFFLE_SAMPLES=True,
    BEAM_WIDTH=1,
    
    SEED = 0,)

args_ctc = Namespace(  
    SEED = 0,
    FREEZE_ENCODER=False,
    DROPOUT=0.2,
    WARMUP_PERIOD=1000,
    
    VAL_DF = None,
    PRINT_SAMPLE_PS = True,
    
    # parameters for encoder/decoder CTC
    DIM_ENCODER=256,
    DIM_CONTEXT=256,
    KEEP_HIDDEN_VECTOR=False,
    N_LEVELS_CONTEXT=1,
    CONTEXT_RNN="LSTM", # "RNN" or "GRU"
    #N_PREDICTIONS=12,
    N_NEGATIVE_SAMPLE =128,

    # data parameters
    SIZE_WINDOW = 20480,
    DATA_EXT='.wav',
    SOURCE_FRAME_RATE=48000,
    DEST_FRAME_RATE=16000,

    # training parameters
    PATIENCE=10, # early stopping limit
    N_EPOCH=150,
    CHECKPOINT_PATH = 'checkpoint_data/checkpoint_30.pt',
    MIN_LR = 0.0,
    LEARNING_RATE = 1e-3, # set starting lr
    OPTIMIZER=torch.optim.Adam,
    SCHEDULER=torch.optim.lr_scheduler.CosineAnnealingLR,
    WEIGHT_DECAY = 0.000,
    BEAM_WIDTH=1, # for beam search, set to 1 since we are using max_decoding
    

    # dataloader parameters
    PATH_TRAIN_DATA_CER = "content/clips_16k/",
    PATH_VAL_DATA_CER = "content/val/",
    PATH_TEST_DATA_CER = "content/test",
    PATH_LETTER_DATA_CER = 'content/char_to_labels.txt',
    PATH_PSEUDOLABEL_DATA_CER = 'content/char_to_labels_ps.txt',

    TRAIN_BATCH_SIZE=16,
    VAL_BATCH_SIZE = 8,
    MAX_TRAINING_DURATION = 2*60*60, # change between 2, 10 and 20 hrs

    AUDIO_PATH_TRAIN="content/clips_16k",
    AUDIO_PATH_DEV = "content/val/",
    AUDIO_PATH_TEST = "content/test/",

    FINAL_MODEL_SAVE_PATH = '/content/drive/My Drive/Colab Notebooks/data/fr/',
    CHECKPOINT_SAVE_PATH = "/content/checkpoint.ckpt",

    CHARS =["^", "a", "b", "c" , "d" ,"e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", 
            "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", " "], #replace with  your alphabet
)

args_ctc.DEVICE = True if torch.cuda.is_available() else False
args_ctc.CUT_OFF_TOP_N = len(args_ctc.CHARS)