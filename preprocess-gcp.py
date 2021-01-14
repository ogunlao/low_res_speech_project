import wget
import pandas as pd
from patoolib import extract_archive

import os
from pydub import AudioSegment as am
import wave
import contextlib
import time
import librosa

from unicodedata import normalize
import string
import re
import shutil
try:
    from utils import make_dirs
except:
    from .utils import make_dirs
    
import multiprocessing
import pandas as pd


def get_audio_duration(df, df_name, src_path, 
                   dest_path='content/clips_16k/',
                   dest_frame_rate=16000):
  
    make_dirs(dest_path)

    t0 = time.time()

    df = df.copy()
    df['duration'] = 0.0

    for i in range(len(df)):
      mp3_file = df.iloc[i].path
      src = os.path.join(src_path, mp3_file)
      
      # convert mp3 to wav                                                            
      sound = am.from_mp3(src)
            
      # calculate duration of wav file
      duration = sound.duration_seconds
      
      #duration = librosa.core.get_duration(filename=dst)

      df.at[i, 'duration'] = float(duration)
    df.to_csv(df_name+'.csv', index=False)

    print(f'Resampling for {df_name} finished in {time.time() - t0} seconds')

if __name__ == '__main__':
    
    starttime = time.time()
    processes = []
    
    # Get data frame to sample from
    audio_dir = "cv-corpus-6.1-2020-12-11/rw/" # replace with where audio is downloaded

    train_df = pd.read_csv(os.path.join(audio_dir, "train.tsv"), sep='\t')
    # val_df = pd.read_csv(os.path.join(audio_dir, "dev.tsv"), sep='\t')
    # test_df = pd.read_csv(os.path.join(audio_dir, "test.tsv"), sep='\t')
    
    # split large train set for multiple processes
    
    train_df = train_df[train_df['gender'].notnull()]
    
    num_cpu = multiprocessing.cpu_count()    
    
    size_split = len(train_df)//num_cpu
    
    for idx in range(num_cpu):
        df_name = 'train_df'+str(idx)
        
        if idx < num_cpu-1:
            p = multiprocessing.Process(target=get_audio_duration,
                                        args=(train_df[idx*size_split : (idx+1)*size_split].copy(), df_name, 
                                            "/content/cv-corpus-6.1-2020-12-11/rw/clips/",
                                                "content/clips_16k/", 16000,))
        else:
            p = multiprocessing.Process(target=get_audio_duration,
                                        args=(train_df[idx*size_split :].copy(), df_name, 
                                            "/content/cv-corpus-6.1-2020-12-11/rw/clips/",
                                                "content/clips_16k/", 16000,))

        processes.append(p)
        p.start()
        
    for process in processes:
        process.join()
    
    print('That took {} seconds'.format(time.time() - starttime))

