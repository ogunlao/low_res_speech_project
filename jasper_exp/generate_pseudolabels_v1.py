# NeMo's "core" package
import nemo
import nemo.collections.asr as nemo_asr

import os
import time
from pathlib import Path
import pandas as pd

from args_nemo import args

from build_manifest import build_manifest_ps

from phonemizer import phonemize
from phonemizer import separator


def get_pseudolabel_n_phonemize(model, df, 
                        wav_dir, separator, 
                        save_file_name, bs=32):
    t0 = time.time()
    df = df.copy()

    transcriptions = []
    for i in range(0, len(df), bs):
        # get filename

        batch_paths = df[i:i+bs].path.values
        paths = []
        for path in batch_paths:
            curr_path = os.path.join(wav_dir, path)
            paths.append(curr_path)

        file_name = paths

        transcription = model.transcribe(paths2audio_files=paths, batch_size=bs)
        transcriptions.extend(transcription)

    df['pseudolabel'] = transcriptions
    print("finished pseudolabelling")

    # TODO check if string is not nan
    prev_len = len(df)
    df = df[df['pseudolabel'].notna()]
    df = df.reset_index(drop=True)
    
    if prev_len != len(df):
      print(f"{prev_len - len(df)} pseudolabels dropped cos on na")
      
    # TODO: uncomment when converting pseudolabels to phonemes
    
    df_columns = list(df.columns.values)
    df_array = []
    for i, data in df.iterrows():
        transcription = data.pseudolabel
        phoneme = phonemize(transcription, 
                            backend='festival', 
                            separator=separator, njobs=6)
        if phoneme:
            data['phoneme'] = phoneme
            df_array.append(data)

        if i%100 == 0:
            print('Now at index', i)

    new_df = pd.DataFrame(df_array, columns=df_columns+['phoneme'])
    
    # assert len(phonemes) == len(transcriptions)
    # df['phoneme'] = phonemes
    # df.to_csv(data_path+os.sep+save_file_name, index=False)
    
    new_df.to_csv(data_path+os.sep+save_file_name, index=False)
    # if i%10 == 0:
    #   print('Now at index', i)
    print(f'Code finished in {time.time() - t0} seconds')
    return new_df

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
    asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=args.get('MODEL_NAME'))

    curr_path = Path(__file__).parent.absolute()
    data_path = str(curr_path)+os.sep+'..'+os.sep+'..'+os.sep+args.get('DATA_FOLDER')
    data_sample_dir = os.path.join(data_path, args.get('SAMPLED_DATA_FOLDER'))

    phoneme_sep = separator.Separator(word=' ', syllable='', phone='-') #separate each phoneme in the audio

    train_ps = pd.read_csv(os.path.join(data_path, args.get('TRAIN_PS_CSV')))
    train_ps = get_pseudolabel_n_phonemize(asr_model, train_ps, 
                        data_sample_dir,
                        phoneme_sep, 
                        args.get('TRAIN_W_PS_CSV'), bs=32)

    if args.get('VAL_PS_CSV'):
        val_ps = pd.read_csv(os.path.join(data_path, args.get('VAL_PS_CSV')))
        val_ps = get_pseudolabel_n_phonemize(asr_model, val_ps, 
                        data_sample_dir,
                        phoneme_sep, 
                        args.get('VAL_W_PS_CSV'), bs=32)
    # TODO uncomment to generate the manifest
    # phoneme_set, phoneme_map = extract_phonemes_n_map(train_ps)
    # print('Vocab_Train: ', phoneme_map.values(), 'vocab_len_Train: ', len(list(phoneme_map.values())))


    #print(f'Total no. of phonemes {len(phoneme_set)}')

    # Building Manifests
    print("******")

    # train_manifest = os.path.join(data_path, 'train_manifest.json')
    # val_manifest = os.path.join(data_path, 'val_manifest.json')

    # #if not os.path.isfile(train_manifest):    
    # build_manifest_ps(train_ps, 
    #             train_manifest,
    #             data_sample_dir,
    #             phoneme_map,)

    # build_manifest_ps(val_ps, 
    #             val_manifest,
    #             data_sample_dir,
    #             phoneme_map,) 

    # print("Training and val manifest created.")

    # print("***Done***")






