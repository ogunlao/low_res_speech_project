import nemo
import nemo.collections.asr as nemo_asr

from ruamel.yaml import YAML

from pathlib import Path
import os

from omegaconf import DictConfig
from omegaconf import OmegaConf
import argparse
import pandas as pd

from configs.args_nemo_finetune import args as args_default

from helpers.eval_model import eval_model

import sys
import copy

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


def collate_args(args1, args2):
    args = {**args2, **args1}
    return args    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-file',
                        help="Checkpoint file .ckpt of phoneme pretrained model to use for finetuning")

    args = parser.parse_args()
    print(vars(args))
    args = collate_args(args1=vars(args), 
                        args2=vars(args_default)
                        )
    curr_path = Path(__file__).parent.absolute()
    config_path = str(curr_path)+os.sep+'..'+os.sep+'..'+os.sep+args.get('CONFIG_PATH')
    data_path = str(curr_path)+os.sep+'..'+os.sep+'..'+os.sep+args.get('DATA_FOLDER')
    model_path = str(curr_path)+os.sep+'..'+os.sep+'..'+os.sep+args.get('SAV_CHECKPOINT_PATH')
   

    # Load jasper parameters
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)

    model_full_path = os.path.join(model_path, args.get("checkpoint_file"))
    asr_model = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(model_full_path)

    train_manifest = os.path.join(data_path, args.get("TRAIN_MANIFEST"))
    val_manifest = os.path.join(data_path, args.get("VAL_MANIFEST"))
    test_manifest = os.path.join(data_path, args.get("TEST_MANIFEST"))

    params['model']['train_ds']['manifest_filepath'] = train_manifest
    params['model']['validation_ds']['manifest_filepath'] = val_manifest
    
    args['curr_path'] = curr_path
    args['data_path'] = data_path
    
    train_cer = eval_model(asr_model, params, args, train=True, valid=False)      
    val_cer = eval_model(asr_model, params, args, valid=True)
    
    if args.get("TEST_MANIFEST"):
        test_cer = eval_model(asr_model, params, args,  
                    manifest_file=test_manifest,
                    valid=False, test=True,)
    
    print(f'train cer: {train_cer}')
    print(f'val cer: {val_cer}')
    print(f'test cer: {test_cer}')
        
    with open('result.txt', "a+") as f:
        tag = args.get("EXPERIMENT_TAG")
        print(f'results for model training {tag}', file=f)
        print(f'train cer: {train_cer}', file=f)
        print(f'val cer: {val_cer}', file=f)
        print(f'test cer: {test_cer}', file=f)
        print(f'------------------------------------------', file=f)
