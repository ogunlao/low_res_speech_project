# --- Building Manifest Files --- #
import json
import os
import librosa
import pandas as pd

from unicodedata import normalize
import string
import re

def map_transcript(transcript, phoneme_map):
    transcript = transcript.replace(' ', 'sil-').split('-')
    temp = []

    for phoneme in transcript:
        if phoneme == '':
            pass
        else:
            temp.append(phoneme_map[phoneme])
        
    return ''.join(temp[:-1])

# function to clean tet, remove punctuations and normalize text
def clean_sentence(sentence):
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
    #table = str.maketrans('', '', string.punctuation)

    # normalize unicode characters
    sentence = normalize('NFD', sentence).encode('ascii', 'ignore')
    sentence = sentence.decode('UTF-8')
    # tokenize on white space
    sentence = sentence.split()
    # convert to lower case
    sentence = [word.lower() for word in sentence]
    # remove punctuation from each token   
    sentence = [word.translate(table) for word in sentence]
    # remove non-printable chars form each token
    sentence = [re_print.sub('', w) for w in sentence]
    # remove tokens with numbers in them
    # sentence = [word for word in sentence if word.isalpha()]
    # return as string

    return ' '.join(sentence)

# Function to build a manifest
def build_manifest_ps(df, manifest_path, data_dir, phoneme_map, hours=None):
    with open(manifest_path, 'w') as fout:
        # Lines look like this:
        total = 0.0
        for i, data in df.iterrows():
            # Get transcripts - phoneme
            transcript = data.phoneme
            transcript = map_transcript(transcript, phoneme_map)

            # Get audio path
            wav_path = data.path
            audio_path = os.path.join(
                data_dir, wav_path,)

            duration = data.duration
            total+=duration
            if hours and total >= hours:
              break
            # Write the metadata to the manifest
            metadata = {
                "audio_filepath": audio_path,
                "duration": float(duration),
                "text": transcript
            }

            json.dump(metadata, fout, ensure_ascii=False)
            fout.write('\n')

def build_manifest(df, manifest_path, data_dir):
    with open(manifest_path, 'w') as fout:
        # Lines look like this:
        for i, data in df.iterrows():
            transcript = data.sentence
            transcript = clean_sentence(transcript)

            try:
                # Get audio path
                wav_path = data.path
                audio_path = os.path.join(
                    data_dir, wav_path,)

                duration = data.duration
            except:
                print('---------------------')
                print('wav_path', wav_path)


            # Write the metadata to the manifest
            metadata = {
                "audio_filepath": audio_path,
                "duration": duration,
                "text": transcript.lower()
            }

            json.dump(metadata, fout, ensure_ascii=False)
            fout.write('\n')