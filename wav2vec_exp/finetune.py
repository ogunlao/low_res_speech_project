from pathlib import Path
import os
import pandas as pd
from collections import defaultdict
import soundfile

from unicodedata import normalize
import string
import re
from args import args

# function to clean tet, remove punctuations and normalize text
def clean_sentence(sentence):
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
    #table = str.maketrans('', '', string.punctuation)

    # normalize unicode characters
    sentence = normalize('NFD', sentence).encode('ascii', 'ignore')
    sentence = sentence.decode('UTF-8')
    # tokenize on white space
    sentence = sentence.split()
    # convert to lower case
    sentence = [word.lower() for word in sentence]
    # remove punctuation from each token   
    sentence = [word.translate(table) for word in sentence]
    # remove non-printable chars form each token
    sentence = [re_print.sub('', w) for w in sentence]
    # remove tokens with numbers in them
    # sentence = [word for word in sentence if word.isalpha()]
    # return as string

    return ' '.join(sentence)

def separate_char(sentence):
    sentence = list(sentence)
    sentence = ' '.join(sentence)
    return sentence

def create_label_file(df, dest_path):
    with open(dest_path, 'w') as f:
        for i, data in df.iterrows():
            sentence = clean_sentence(data.sentence)
            sentence = sentence.replace(' ', '|')
            sentence = separate_char(sentence)
            print(sentence, file=f)
    print(f'Finished writing to {dest_path}')

def generate_char_dict(df):
    char_dict = defaultdict(int)

    for sentence in df['sentence'].values:
        if sentence:
            sentence = clean_sentence(sentence)
            for char in sentence:
                char_dict[char]+=1

    print(char_dict)
    print(len(char_dict))
    
    return char_dict

def save_char_dict(char_dict):
    with open(os.path.join(download_path, 'dict.ltr.txt'), 'w') as f:
        char_tuple = sorted(char_dict.items())
        for char, total in char_tuple:
            if char == ' ': char = '|'
            print(f'{char} {total}', file=f)
    

def create_tsv(df, dir_path, dest_path):
    with open(dest_path, 'w') as f:
        print(dir_path, file=f)
        
        for i, data in df.iterrows():
            file_path = data.path
            frames = soundfile.info(os.path.join(dir_path, file_path)).frames
            print(
                "{}\t{}".format(file_path, frames), file=f
            )
    print(f'Finished writing to {dest_path}')   
    
    
if __name__ == '__main__':
    args = vars(args)

    curr_path = Path(__file__).parent.absolute()
    download_path = str(curr_path)+os.sep+'..'+os.sep+'..'+os.sep+args.get('DATA_FOLDER')
    sample_dest_path = download_path+os.sep+args.get('SAMPLED_DATA_FOLDER')
    
    train_path = os.path.join(download_path, args.get('FINETUNE_CSV'))
    train_df = pd.read_csv(train_path)

    val_path = os.path.join(download_path, args.get('VAL_CSV'))
    val_df = pd.read_csv(val_path)
    
    test_path = os.path.join(download_path, args.get('TEST_CSV'))
    test_df = pd.read_csv(val_path)
    
    # train_df.to_csv(os.path.join(download_path, args.get('FINETUNE_TSV')), sep='\t')
    # val_df.to_csv(os.path.join(download_path, args.get('VAL_TSV')), sep='\t')
    # test_df.to_csv(os.path.join(download_path, args.get('TEST_TSV')), sep='\t')

    char_dict = generate_char_dict(train_df)
    save_char_dict(char_dict)
    
    create_label_file(df=train_df, 
                      dest_path=os.path.join(download_path, 'train.ltr'))

    create_label_file(df=val_df, 
                      dest_path=os.path.join(download_path, 'dev_other.ltr'))

    create_label_file(df=test_df, 
                      dest_path=os.path.join(download_path, 'valid.ltr'))
    
   
    train_path = os.path.join(download_path, args.get('FINETUNE_CSV'))
    train = pd.read_csv(train_path)

    val_path = os.path.join(download_path, args.get('VAL_CSV'))
    val = pd.read_csv(val_path)

    test_path = os.path.join(download_path, args.get('TEST_CSV'))
    test = pd.read_csv(test_path)

    train_dest_path = os.path.join(download_path, args.get('FINETUNE_TSV'))
    val_dest_path = os.path.join(download_path, args.get('VAL_TSV'))
    test_dest_path = os.path.join(download_path, args.get('TEST_TSV'))

             
    create_tsv(df=train, 
                dir_path=sample_dest_path,
                dest_path=train_dest_path)

    create_tsv(df=val, 
                dir_path=sample_dest_path,
                dest_path=val_dest_path)

    create_tsv(df=test, 
                dir_path=sample_dest_path,
                dest_path=test_dest_path)
    
## Run this to finetune

# fairseq-hydra-train \
#     task.data=/content/weak_supervision/data/ \
#     distributed_training.distributed_world_size=1 \
#     optimization.update_freq='[128]' \
#     optimization.max_epoch=2 \
#     model.w2v_path=/content/weak_supervision/models/checkpoint_best.pt \
#     task.normalize=True \
#     checkpoint.best_checkpoint_metric="loss" \
#     --config-dir /content/weak_supervision/low_res_speech_project/wav2vec_exp/fairseq/examples/wav2vec/config/finetuning \
#     --config-name base_100h 