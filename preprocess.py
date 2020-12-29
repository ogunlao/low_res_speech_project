import wget
import pandas as pd
from pathlib import Path
from patoolib import extract_archive

import os
from pydub import AudioSegment as am
import wave
import contextlib
import time
import librosa

from .utils import make_dirs
from .utils import args_proc as args

def dl_commonvoice_data(url, save_path=None, unpack=True):
    r"""
    Download common voice file from mozilla commonvoice repo
    
    usage
    -----
    >> url = "https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-6.1-2020-12-11/pa-IN.tar.gz"
    >> dl_commonvoice_data(url, unpack=True)
    """
    if save_path is None:
        save_path = str(Path.cwd())
    filename = wget.download(url, save_path)
    print(filename)
    
    print(f"file sucessfully downloaded in dir {save_path}")
    if unpack:
        patoolib.extract_archive(filename, outdir=save_path)
        print(f"file sucessfully unpacked in dir {save_path}")


def convert_to_wav(mp3_file, 
                   src_path, 
                   dest_path='/content/files/',
                   dest_frame_rate=16000):
    src = os.path.join(src_path, mp3_file)
    dst = os.path.join(dest_path, mp3_file[:-4]+'.wav')

    # convert mp3 to wav                                                            
    sound = am.from_mp3(src)
    sound = sound.set_frame_rate(dest_frame_rate)
    sound.export(dst, format="wav")

    return mp3_file[:-4] + '.wav'


def get_samples(data_dict,
                audio_src_path,
                wav_sav_path,
                seed=args.SEED, 
                shuffle=args.SHUFFLE_SAMPLES,):
    r"""
    Generate different samples for training from the same dataser. Also, subsamples the audio data and convert file format
    max_samples: Max duration of different splits in the same dataset
    
    usage
    ------
    data_dict = {
    'train': [train_df, (40, 20)], # in hrs
    'val': [val_df, (2,)],
    'test': [test_df, (2,)] 
    }
    
    >> get_samples(data_dict,
            audio_src_path="/content/cv-corpus-6.1-2020-12-11/pa-IN/clips",
            wav_sav_path="/content/train/", 
            seed=0, 
            shuffle=True)
    """
    
    make_dirs(wav_sav_path)
    
    for data_array in data_dict:
        df, max_samples = data_array:
        # measure wall time   
        t0 = time.time()

        if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True) #shuffle df
        
        df = df.copy()
        df['duration'] = 0.0

        start_idx = 0
        for sample_idx in range(len(max_samples)):
            total = 0.0
            max_duration = max_samples[sample_idx]*3600

            for i in range(start_idx, len(df)):
                mp3_file = df.iloc[i].path
                
                # convert to wav file from mp3
                wav_file = convert_to_wav(mp3_file=mp3_file,
                                            src_path=audio_src_path, 
                                            dest_path=wav_sav_path,
                                            dest_frame_rate=16000)

                # calculate duration of wav file
                audio_path = os.path.join(wav_sav_path, wav_file)
                duration = librosa.core.get_duration(filename=audio_path)

                df.at[i, 'path'] = wav_file
                df.at[i, 'duration'] = float(duration)

                total+=duration
                # if i%1000 == 0:
                #   print("Now at", i)

                if total >= max_duration:
                    temp_df = df[start_idx: i+1]
                        
                    df_save_path = os.path.join(os.getcwd(),  
                                                data_array + str(total//3600) + 'hrs_' + str(sample_idx)+'.csv')
                    
                    temp_df.to_csv(df_save_path, index=False)
                    start_idx = i+1
                    
                    print(f'finished sampling for {total/3600} hrs, csv file saved in {df_save_path}')
                    total = 0.0
                    break

        print(f'Code for {data_array} finished in {time.time() - t0} seconds')