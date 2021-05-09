#!/bin/bash

pip3 install wget
pip3 install patool
pip3 install pydub
# pip3 install torchaudio==0.7.0
pip3 install soundfile
# pip install torchtext==0.8.0 torch==1.7.1 pytorch-lightning==1.2.1
pip3 install nemo-toolkit[all]==1.0.0b1
sudo apt-get install festival espeak-ng #mbrola
pip3 install phonemizer
pip3 install pytorch-lightning==0.9.0
# CUDA 10.1
# pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install torch==1.7.1
pip3 install torchtext==0.8.0



sudo apt-get install libsndfile1-dev
pip install pip==20.1.1