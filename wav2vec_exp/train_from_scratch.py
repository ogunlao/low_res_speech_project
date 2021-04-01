import soundfile
from pathlib import Path
import os
import pandas as pd
from .args import args


args = vars(args)

curr_path = Path(__file__).parent.absolute()
download_path = str(curr_path)+os.sep+'..'+os.sep+args.get('DATA_FOLDER')
sample_dest_path = download_path+os.sep+args.get('SAMPLED_DATA_FOLDER')


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
    

train_path = os.path.join(download_path, args.get('TRAIN_PS_CSV'))
train_df = pd.read_csv(train_path)

val_path = os.path.join(download_path, args.get('TRAIN_PS_CSV'))
val_df = pd.read_csv(val_path)

train_dest_path = os.path.join(download_path, args.get('TRAIN_PS_CSV'))
val_dest_path = os.path.join(download_path, args.get('VAL_PS_CSV'))


if __name__ == '__main__':                 
    create_tsv(df=train_df, 
                dir_path=sample_dest_path,
                dest_path=train_dest_path)

    create_tsv(df=val_df, 
                dir_path=sample_dest_path,
                dest_path=val_dest_path)

## Run script in cmd
  
# fairseq-hydra-train \
#     task.data=~/data/clips_16k/ \
#     distributed_training.distributed_world_size=1 \
#     optimization.update_freq='[128]' \
#     --config-dir ~/fairseq/examples/wav2vec/config/pretraining \
#     --config-name wav2vec2_large_librivox
#     --checkpoint.save-dir=~/model