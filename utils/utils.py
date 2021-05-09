import os
from argparse import Namespace
from unicodedata import normalize
from collections import defaultdict
import torch

def make_dirs(path):
    if not os.path.exists(path):
      os.makedirs(path)

def collate_args(args1, args2):
    args1 = {key: index for key, index in args1.items()}
    args2 = {key: index for key, index in args2.items()}
    args = {**args2, **args1}
    return args
      
def calc_distr(df, punct_set):
    chars = defaultdict(int)
    for sentence in df.sentence.values:
        sentence = normalize('NFD', sentence).encode('ascii', 'ignore')
        sentence = sentence.decode('UTF-8')
        for char in sentence:
            if char not in punct_set:
                if char == " ":
                    chars["<sep>"]+=1
                else: chars[char.lower()]+=1

    #chars_sorted = sorted(chars.items())
    #print(chars_sorted)

    return chars