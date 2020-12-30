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

from unicodedata import normalize
import string
import re
import shutil

from .utils import make_dirs
from .utils import args_preproc as args
from .utils import args_ctc
import progressbar

device = torch.device("cuda:0" if args.DEVICE else "cpu")

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
        extract_archive(filename, outdir=save_path)
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
            wav_sav_path="content/data/", 
            seed=0, 
            shuffle=True)
    """
    
    make_dirs(wav_sav_path)
    
    for data_split in data_dict:
        df, max_samples = data_dict[data_split]
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
            
            sav_path = os.path.join(wav_sav_path, data_split, str(sample_idx))
            make_dirs(sav_path)
            
            for i in range(start_idx, len(df)):
                mp3_file = df.iloc[i].path
                
                # convert to wav file from mp3
                wav_file = convert_to_wav(mp3_file=mp3_file,
                                            src_path=audio_src_path, 
                                            dest_path=sav_path,
                                            dest_frame_rate=16000)

                # calculate duration of wav file
                audio_path = os.path.join(sav_path, wav_file)
                duration = librosa.core.get_duration(filename=audio_path)

                df.at[i, 'path'] = wav_file
                df.at[i, 'duration'] = float(duration)

                total+=duration
                # if i%1000 == 0:
                #   print("Now at", i)

                if total >= max_duration:
                    temp_df = df[start_idx: i+1]
                        
                    df_save_path = os.path.join(os.getcwd(),  
                                                data_split + str(total//3600) + 'hrs_' + str(sample_idx)+'.csv')
                    
                    temp_df.to_csv(df_save_path, index=False)
                    start_idx = i+1
                    
                    print(f'finished sampling for {total/3600} hrs, csv file saved in {df_save_path} and audio saved in {sav_path}')
                    total = 0.0
                    break

        print(f'Code for {data_split} finished in {time.time() - t0} seconds')
        

def clean_sentence(sentence):
    """function to clean text, remove punctuations and normalize text """
    # prepare regex for char filtering
    #re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
    #table = str.maketrans('', '', string.punctuation)

    # normalize unicode characters
    #sentence = normalize('NFD', sentence).encode('ascii', 'ignore')
    #sentence = sentence.decode('UTF-8')
    # tokenize on white space
    sentence = sentence.split()
    # convert to lower case
    sentence = [word.lower() for word in sentence]
    # remove punctuation from each token   
    sentence = [word.translate(table) for word in sentence]
    # remove non-printable chars form each token
    #sentence = [re_print.sub('', w) for w in sentence]
    # remove tokens with numbers in them
    # sentence = [word for word in sentence if word.isalpha()]
    # return as string

    return ' '.join(sentence)


def convert_text_to_index(df, 
                          character_to_index, 
                          audio_path, file_name, 
                          dest_path='', max_sec=None, use_pseudolabel=False):
    if max_sec:
        make_dirs(dest_path)

    total_sec = 0.0
    with open(file_name, "a+") as all_session_text:
        for i in range(len(df)):
            wav_name = df.iloc[i]['path']
            wav_path = os.path.join(audio_path, wav_name)

            if use_pseudolabel:
                sentence = df.iloc[i]['pseudolabels']
            else:
                sentence = clean_sentence(df.iloc[i]['sentence'])
            
            indices = ''
            for c in sentence: 
                indices+=str(character_to_index[c.lower()]) + ' ' 
            #wav_name[:-4] to remove the extension   
            all_session_text.writelines(wav_name[:-4] + ' ' + indices + '\n')
            total_sec += df.iloc[i].duration

            if max_sec: # move to new path if subsampling
              shutil.copy2(wav_path, dest_path)
              # if os.path.exists(wav_path):
              #   os.rename(wav_path, os.path.join(dest_path, wav_name))

            if max_sec and total_sec >= max_sec:
                break
        
    print(f'Total duration of file added from {audio_path} is {total_sec//3600}hrs')
    

def get_pseudolabels(df, data_dataloader,
                     cpc_model,
                     character_classifier, args=args):
  df = df.sort_values(by=['path'], axis=0, ascending=True,
                      inplace=False, ignore_index=True)

  t0 = time.time()

  downsampling_factor = 1
  cpc_model.eval()
  character_classifier.eval()

  all_transcriptions = []

  print("Starting the get pesudolabels using max-decoding")
  bar = progressbar.ProgressBar(maxval=len(data_dataloader))
  bar.start()

  decoder = CTCBeamDecoder(args_ctc.CHARS, log_probs_input=False, blank_id=0, beam_width=args_ctc.BEAM_WIDTH,
                           cutoff_top_n=args_ctc.CUT_OFF_TOP_N)  # model_path="/content/wiki_00.lm.arpa")
  for index, data in enumerate(data_dataloader):

    bar.update(index)
    with torch.no_grad():
        seq, sizeSeq, phone, sizePhone = prepare_data(data)
        x_batch_len = seq.shape[-1]
        c_feature, _, _ = cpc_model(seq.to(device), phone.to(device))
        bs = c_feature.size(0)
        sizeSeq = sizeSeq / downsampling_factor
        predictions = torch.nn.functional.softmax(
            character_classifier(c_feature), dim=2
        ).cpu()
        phone = phone.cpu()
        sizeSeq = sizeSeq.cpu()
        sizePhone = sizePhone.cpu()
        # print("predictions",predictions.argmax(2)[0])
        # print(phone[0])

        seq_len = torch.tensor([int(predictions.shape[1]*sizeSeq[i]/(x_batch_len))
                                for i in range(predictions.shape[0])])  # this is an approximation, should be good enough
        #print(seq_len)

        output, scores, timesteps, out_seq_len = decoder.decode(
            predictions, seq_lens=seq_len)

        output = output[torch.arange(bs), scores.argmax(1), :]
        out_seq_len = out_seq_len[torch.arange(bs), scores.argmax(1)]

        # print(output, output.shape)
        # print(out_seq_len, out_seq_len.shape)

        batch_size = output.shape[0]

        transcripts = ["".join(args.CHARS[n] for n in output[bs]
                               [:out_seq_len[bs]]) for bs in range(batch_size)]

        all_transcriptions.extend(transcripts)
  df['pseudolabels'] = all_transcriptions
  bar.finish()
  return df
