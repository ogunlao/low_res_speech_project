import os
import shutil
import pandas as pd

def make_dirs(path):
    if not os.path.exists(path):
      os.makedirs(path)

def move_files(df, curr_path, new_path):
    make_dirs(new_path)
    for i, data in df.iterrows():
        old_dir = os.path.join(curr_path, data.path)
        new_dir = os.path.join(new_path, data.path)
        shutil.copy(old_dir, new_dir)
    print('All files moved')
    

if __name__ == '__main__':
    df_dir = ''
    df = pd.read_csv(df_dir)
    curr_path = ''
    new_path = ''
    move_files(df, curr_path, new_path)
        
        
     