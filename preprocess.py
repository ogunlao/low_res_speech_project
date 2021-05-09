import pandas as pd
from pathlib import Path
import argparse
import time
import glob
import os

from configs import args_prep as args_default
from utils import make_dirs
from utils import collate_args
from utils import dl_commonvoice_data

from utils import sample_audio

def combine_duration_files(args):
    
    save_duration_path = args.get('DURATION_SAV_FILE', '') 
    all_filenames = glob.glob(download_path+os.sep+save_duration_path+"*.csv")

    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    #export to csv
    combined_csv.to_csv(download_path+os.sep+save_duration_path, index=False, encoding='utf-8-sig')
    print(f"Duration csv combined and saved in {download_path} as {save_duration_path}")
                
 
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="url path of the commonvoice dataset")
    args = parser.parse_args()

    args = collate_args(args1=vars(args), 
                        args2=vars(args_default)
                        )

    starttime = time.time()
    
    curr_path = Path(__file__).parent.absolute()
    download_path = str(curr_path)+os.sep+'..'+os.sep+args.get('DATA_FOLDER')
    make_dirs(download_path)

    # 1) Download files from commonvoice if they do not exis
    if args.get('url'):
        dl_commonvoice_data(download_path, args)
    
    # 2) Downsample files in folder to 16k sampling rate and save duration of each file
    try:
        sample_audio(args)
    except:
        print("Audio files are not found for sampling")
    
    # 3) Make single duration file. This 
    combine_duration_files(args)

    print('That took {} seconds'.format(time.time() - starttime))
