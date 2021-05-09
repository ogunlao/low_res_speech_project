from pathlib import Path
import os

import pandas as pd

from configs import args_prep as args_default
from utils import collate_args

def create_train_finetune_split(train_duration, 
                                finetune_duration, 
                                validation_duration, args):

    curr_path = Path(__file__).parent.absolute()
    data_path = str(curr_path)+os.sep+'..'+os.sep+args.get('DATA_FOLDER')
    raw_audio_paths = args.get('RAW_AUDIO_PATH')

    train_df = pd.read_csv(
        os.path.join(data_path, 'train_resampled.csv'))
 

    columns_to_select = ['path', 'sentence', 'gender', 'duration']
    train_df = train_df[columns_to_select]

    train_df = train_df.sample(frac=1, 
                        random_state=args.get('SEED', 1234)).reset_index(drop=True)

    total_duration = 0.0
    current_idx = 0
    
    split_details = {'train': [train_duration, args.get('TRAIN_PS_CSV')],
                     'finetune': [finetune_duration, args.get('FINETUNE_CSV')],
                     'val': [validation_duration, args.get('VAL_PS_CSV')]
                     }
    
    df = train_df.copy(deep=True)
    last_row_idx = 0
    for split in split_details:
        if not split_details[split][0]: 
            continue
        samples_duration = split_details[split][0]
        path_name = split_details[split][1]

        for row in df.itertuples():
            if total_duration <= samples_duration:
                total_duration += float(row.duration)
                last_row_idx = row.Index
            else:
                break
            
        temp_df = df[0:last_row_idx].copy()
        temp_df.to_csv(data_path + os.sep + path_name, index=False)
        print(f'{total_duration/3600}hrs {split} split done')
        
        # reset
        df = df[last_row_idx:].reset_index(drop=True).copy()
        last_row_idx = 0
        total_duration = 0
        temp_df = None
        

if __name__ == '__main__':
    args = collate_args(args1={}, 
                        args2=vars(args_default)
                        )
    create_train_finetune_split(args.get('TRAIN_DURATION'),
                                args.get('FINETUNE_DURATION'),
                                args.get('VALIDATION_DURATION'), 
                                args,
                                )