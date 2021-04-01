from pathlib import Path
import os
import pandas as pd

from unicodedata import normalize
import string
import re

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


def create_label_file(df, dest_path):
    with open(dest_path, 'w') as f:
        for i, data in df.iterrows():
            sentence = clean_sentence(data.sentence)
            sentence = sentence.replace(' ', '|')

            print(sentence, file=f)
    print(f'Finished writing to {dest_path}')

    
if __name__ == '__main__':
    
    args = vars(args)

    curr_path = Path(__file__).parent.absolute()
    download_path = str(curr_path)+os.sep+'..'+os.sep+args.get('DATA_FOLDER')
    sample_dest_path = download_path+os.sep+args.get('SAMPLED_DATA_FOLDER')
    
    train_path = os.path.join(download_path, args.get('FINETUNE_CSV'))
    train_df = pd.read_csv(train_path)

    val_path = os.path.join(download_path, args.get('VAL_CSV'))
    val_df = pd.read_csv(val_path)
    
    test_path = os.path.join(download_path, args.get('TEST_CSV'))
    test_df = pd.read_csv(val_path)

    train_dest_path = os.path.join(download_path, args.get('TRAIN_PS_CSV'))
    val_dest_path = os.path.join(download_path, args.get('VAL_PS_CSV'))
    
   
    create_label_file(df=train_df, 
                      dest_path=os.path.join(download_path, 'train.ltr'))

    create_label_file(df=val_df, 
                      dest_path=os.path.join(download_path, 'valid.ltr'))

    create_label_file(df=test_df, 
                      dest_path=os.path.join(download_path, 'test.ltr'))
    
    
## Run this to finetune

# !fairseq-hydra-train \
#     task.data=~/data/clips_16k/ \
#     distributed_training.distributed_world_size=1 \
#     optimization.update_freq='[64]' \
#     model.w2v_path='~/model/wav2vec_small.pt' \
#     --config-dir ~/fairseq/examples/wav2vec/config/finetuning \
#     --config-name base_100h