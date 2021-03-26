# NeMo's "core" package
import nemo
import nemo.collections.asr as nemo_asr

import os
import time
from pathlib import Path
import pandas as pd

from args_nemo import args

from build_manifest import build_manifest_ps


def get_pseudolabels(model, df, 
                        wav_dir, 
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

    df.to_csv(data_path+os.sep+save_file_name, index=False)
    # if i%10 == 0:
    #   print('Now at index', i)
    print(f'Code finished in {time.time() - t0} seconds')
    return df

if __name__ == "__main__":
    args = vars(args)
    asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=args.get('MODEL_NAME'))

    curr_path = Path(__file__).parent.absolute()
    data_path = str(curr_path)+os.sep+'..'+os.sep+'..'+os.sep+args.get('DATA_FOLDER')
    data_sample_dir = os.path.join(data_path, args.get('SAMPLED_DATA_FOLDER'))


    train_ps = pd.read_csv(os.path.join(data_path, args.get('TRAIN_PS_CSV')))
    train_ps = get_pseudolabels(asr_model, train_ps, 
                        data_sample_dir, 
                        args.get('TRAIN_W_PS_CSV'), bs=32)

    if args.get('VAL_PS_CSV'):
        val_ps = pd.read_csv(os.path.join(data_path, args.get('VAL_PS_CSV')))
        val_ps = get_pseudolabels(asr_model, val_ps, 
                        data_sample_dir,
                        args.get('VAL_W_PS_CSV'), bs=32)

    print("***Done***")






