import wget
import pandas as pd
import numpy as np
import random
from pathlib import Path
import argparse
import time
import glob
import os

from patoolib import extract_archive
from pydub import AudioSegment as am
import concurrent.futures
#import progressbar

from args_file import args as args_default
from utils import make_dirs
from utils import collate_args

def dl_commonvoice_data(download_path, args, unpack=True):
    r"""
    Download common voice file from mozilla commonvoice repo

    usage
    -----
    >> url = "https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-6.1-2020-12-11/pa-IN.tar.gz"
    >> dl_commonvoice_data(download_path, args, unpack=True)
    """
    if not os.path.exists(download_path+os.sep+args.get('RAW_AUDIO_PATH')):
        file_name = download_path+os.sep+'rw.tar.gz'
        if not os.path.isfile(file_name):
            filename = wget.download(args.get('url'), download_path)
            print(f"File sucessfully downloaded in dir {download_path}")

        if unpack:
            extract_archive(file_name, outdir=download_path, verbose=0)
            print(f"file sucessfully unpacked in dir {download_path}")
    else:
        print('File already exists in dir {download_path}')

def get_audio_samples(audio_file, src_path, 
                        dest_path, dest_frame_rate=16000):
    """Samples audio from "wav_file" path with specified frame rate"""

    wav_file = str(Path(audio_file).stem) + ".wav"

    dest_path = os.path.join(dest_path, wav_file)
    
    # convert mp3 to wav
    
    try:                                                          
        sound = am.from_mp3(audio_file)
    
        sound = sound.set_frame_rate(dest_frame_rate)
                
        sound.export(dest_path, format="wav")
    except Exception:
        print(f'File {audio_file} unable to be processed')
        return wav_file, 0

    return wav_file, sound.duration_seconds

def download_n_subsample(args):
    """Download audio files, and resamples to required sampling rate

    Args:
        args (dict): Contains the arguments required for parsing directories
    """
    curr_path = Path(__file__).parent.absolute()
    download_path = str(curr_path)+os.sep+'..'+os.sep+args.get('DATA_FOLDER')
    make_dirs(download_path)

    sample_dest_path = download_path+os.sep+args.get('SAMPLED_DATA_FOLDER')
    if len(os.listdir(sample_dest_path)) != 0: # if processed files not found..
        if args.get('url'):
            dl_commonvoice_data(download_path, args)

        raw_audio_paths = args.get('RAW_AUDIO_PATH')
        audio_files = glob.glob(download_path + os.sep + str(raw_audio_paths) + os.sep + "*.mp3")
        
        sample_source_path = download_path + os.sep + str(raw_audio_paths)
        sample_dest_path = download_path + os.sep + args.get('SAMPLED_DATA_FOLDER')
        make_dirs(sample_dest_path)

        processes = []
        
        # subsample audio files using threads
        with concurrent.futures.ThreadPoolExecutor() as executor:
            all_futures = [executor.submit(get_audio_samples, 
                                        audio_file, sample_source_path,
                                            sample_dest_path, 16000) for audio_file in audio_files]

        results = [f.result() for f in all_futures]

        # save audio path and duration in file
        if not os.path.exists(
            os.path.join(download_path, args.get('DURATION_SAV_FILE'))
            ):
            print(f"Writing duration info to file {args.get('DURATION_SAV_FILE')}")

            with open(os.path.join(download_path, args.get('DURATION_SAV_FILE')), "w+") as f:
                for path, duration in results:
                    f.write(f"{path} {duration}")
                    f.write("\n")

def create_train_finetune_split(train_duration, 
                                finetune_duration, 
                                validation_duration, args):
    curr_path = Path(__file__).parent.absolute()
    data_path = str(curr_path)+os.sep+'..'+os.sep+args.get('DATA_FOLDER')
    raw_audio_paths = args.get('RAW_AUDIO_PATH')

    train_df = pd.read_csv(
        os.path.join(data_path, raw_audio_paths, '..', 'train.tsv'),
        sep='\t')
    clips = {}
    with open(data_path+os.sep+args.get('DURATION_SAV_FILE'), 'r') as f:
        for line in f:
            file_name, duration = line.split()
            clips[file_name] = duration

    train_df['path'] = train_df['path'].apply(
        lambda x: x[:-4]+'.wav')

    train_df['duration'] = train_df['path'].apply(
        lambda x: clips.get(x, 0.0)) # TODO: Change this to zero
    
    columns_to_select = ['path', 'sentence', 'gender', 'duration']
    train_df = train_df[columns_to_select]

    train_df = train_df.sample(frac=1, 
                        random_state=args.get('SEED', 1234)).reset_index(drop=True)

    start_index = 0
    total_duration = 0.0

    train = None
    for i in range(start_index, len(train_df)):
        if total_duration <= train_duration:
            total_duration += float(train_df.iloc[i].duration)
        else:
            train = train_df[start_index:i].copy()
            train.to_csv(data_path+os.sep+args.get('TRAIN_PS_CSV'), index=False)
            print(f'{total_duration/3600}hrs training split done')
            start_index = i
            total_duration = 0
            break

    for i in range(start_index, len(train_df)):
        if total_duration <= finetune_duration:
            total_duration += float(train_df.iloc[i].duration)
        else:
            train = train_df[start_index:i].copy()
            train.to_csv(data_path+os.sep+args.get('FINETUNE_CSV'), index=False)
            print(f'{total_duration/3600}hrs finetune split done')
            start_index = i
            total_duration = 0
            break


    if validation_duration: # validation set for pseudolabel pretraining
        for i in range(start_index, len(train_df)):
            if total_duration <= validation_duration:
                total_duration += float(train_df.iloc[i].duration)
            else:
                train = train_df[start_index:i].copy()
                train.to_csv(data_path+os.sep+args.get('VAL_PS_CSV'), index=False)
                print(f'{total_duration/3600}hrs validation split done')
                start_index = i
                break        
 
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="url path of the commonvoice dataset")
    args = parser.parse_args()

    args = collate_args(args1=vars(args), 
                        args2=vars(args_default)
                        )

    starttime = time.time()
    download_n_subsample(args)
    create_train_finetune_split(args.get('TRAIN_DURATION', 50),
                                args.get('FINETUNE_DURATION', 20),
                                args.get('VALIDATION_DURATION'), 
                                args,
                                )

    print('That took {} seconds'.format(time.time() - starttime))
