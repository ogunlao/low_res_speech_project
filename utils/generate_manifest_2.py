# NeMo's "core" package
#import nemo
#import nemo.collections.asr as nemo_asr

import os
import time 
from pathlib import Path
import pandas as pd

from args_nemo import args

from build_manifest import build_manifest_ps

from phonemizer import phonemize
from phonemizer import separator

import multiprocessing as mp


def generate_phoneme_func(df, save_file_name, index, separator):
    df_array = []
    for i, data in df.iterrows():
        transcription = data.pseudolabel
        phoneme = phonemize(transcription, 
                            backend='festival', 
                            separator=separator,)
        if phoneme:
            data['phoneme'] = phoneme
            df_array.append(data)

        if i%100 == 0:
            print(f'Now at index  {i} for process {index}')

    new_df = pd.DataFrame(df_array)
    df_name = data_path+os.sep+'file'+str(index)+'_'+save_file_name
    new_df.to_csv(df_name, index=False)

    print(f"job {index} has finished")

def generate_phonemes(df, 
                        wav_dir, separator, 
                        save_file_name, bs=32):
    t0 = time.time()
    df = df.copy()
    print(save_file_name)
    # TODO check if string is not nan
    prev_len = len(df)
    df = df[df['pseudolabel'].notna()]
    df = df.reset_index(drop=True).copy()

    df_columns = list(df.columns.values)
    if prev_len != len(df):
      print(f"{prev_len - len(df)} pseudolabels dropped cos on na")

    # transcriptions = df['pseudolabel'].values
    total_split = mp.cpu_count()
    print(f'Total cpu available for job is {total_split}. Splitting job...')

    split_range = len(df)//total_split

    jobs = []
    for i in range(total_split):
        start_range = i*split_range
        end_range = start_range + split_range
        if i != total_split-1:
            df_temp = df[start_range : end_range].copy()
        else:
            df_temp = df[start_range:].copy()

        df_temp = df_temp.reset_index(drop=True)

        p = mp.Process(target=generate_phoneme_func, 
                                args=(df_temp, save_file_name, i, separator,))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    #time.sleep(5)
    # combine dataframes
    all_df = pd.DataFrame(columns=list(df.columns.values))
    for i in range(8):
        path = data_path+os.sep+'file'+str(i)+'_'+save_file_name
        # print(path)
        temp_df = pd.read_csv(path)
        all_df = pd.concat([all_df, temp_df])
        all_df = all_df.reset_index(drop=True).copy()

    # assert len(phonemes) == len(transcriptions)
    # df['phoneme'] = phonemes
    #print('datapath', data_path)
    all_df.to_csv(data_path+os.sep+save_file_name, index=False)
    
    print(f'Code finished in {time.time() - t0} seconds')
    return all_df

def extract_phonemes_n_map(df):
    phoneme_set = set()

    for ps in df['phoneme'].values:
        for phone in ps.split('-'):
            phoneme_set.add(phone.strip())

    # map phonemes to single char
    phoneme_map = {}
    i = 65
    for phoneme in phoneme_set:
        if phoneme != '':
            chhar = chr(i)
            phoneme_map[phoneme] = chhar
            i+=1
            if i == 91:
                i = 97

    phoneme_map['sil'] = ' '

    return phoneme_set, phoneme_map


if __name__ == "__main__":
    args = vars(args)
    #asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=args.get('MODEL_NAME'))

    curr_path = Path(__file__).parent.absolute()
    data_path = str(curr_path)+os.sep+'..'+os.sep+'..'+os.sep+args.get('DATA_FOLDER')
    data_sample_dir = os.path.join(data_path, args.get('SAMPLED_DATA_FOLDER'))

    phoneme_sep = separator.Separator(word=' ', syllable='', phone='-') #separate each phoneme in the audio

    train_ps = pd.read_csv(os.path.join(data_path, args.get('TRAIN_W_PS_CSV')))
    train_ps = generate_phonemes(train_ps, 
                        data_sample_dir,
                        phoneme_sep, 
                        args.get('TRAIN_W_Ph_CSV'), bs=32)

    if args.get('VAL_W_PS_CSV'):
        val_ps = pd.read_csv(os.path.join(data_path, args.get('VAL_W_PS_CSV')))
        val_ps = generate_phonemes(val_ps, 
                        data_sample_dir,
                        phoneme_sep, 
                        args.get('VAL_W_Ph_CSV'), bs=32)
    
    #train_ps = pd.read_csv(os.path.join(data_path, args.get('TRAIN_W_Ph_CSV')))
    #val_ps = pd.read_csv(os.path.join(data_path, args.get('VAL_W_Ph_CSV')))
    
    phoneme_set, phoneme_map = extract_phonemes_n_map(train_ps)
    print('Vocab_Train: ', phoneme_map.values(), 'vocab_len_Train: ', len(list(phoneme_map.values())))


    print(f'Total no. of phonemes {len(phoneme_set)}')

    # Building Manifests
    print("******")

    train_manifest = os.path.join(data_path, 'train_manifest_kb.json')
    val_manifest = os.path.join(data_path, 'val_manifest_kb.json')


    build_manifest_ps(train_ps, 
                train_manifest,
                data_sample_dir,
                phoneme_map,)

    build_manifest_ps(val_ps, 
                val_manifest,
                data_sample_dir,
                phoneme_map,) 

    print("Training and val manifest created.")

    print("***Done***")






