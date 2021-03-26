import wave
import contextlib
import time
import glob
from pathlib import Path
from args_file import args
import os
from utils import collate_args



args = collate_args(args1=vars(args), 
                    args2=vars(args)
                    )
curr_path = Path(__file__).parent.absolute()
download_path = str(curr_path)+os.sep+'..'+os.sep+args.get('DATA_FOLDER')
sampled_path = args.get('SAMPLED_DATA_FOLDER')

root_clips_16k = download_path + os.sep + str(sampled_path) + os.sep
audio_files = os.listdir(root_clips_16k)

with open(os.path.join(download_path, args.get('DURATION_SAV_FILE')), "w+") as f1:

    for fname in audio_files:
        with contextlib.closing(wave.open(root_clips_16k+fname,'r')) as f2:
            frames = f2.getnframes()
            rate = f2.getframerate()
            duration = float(frames) / float(rate)
            f1.write(f"{fname} {duration}")
            f1.write("\n")
