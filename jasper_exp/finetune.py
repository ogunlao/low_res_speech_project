import nemo
import nemo.collections.asr as nemo_asr
import torch

from ruamel.yaml import YAML

from pathlib import Path
import os

from omegaconf import DictConfig
from omegaconf import OmegaConf
import argparse
import pandas as pd

from configs.args_nemo_finetune import args as args_default

import sys
import copy

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping


from helpers.change_vocab import change_vocabulary2

seed_everything(42)

def collate_args(args1, args2):
    args = {**args2, **args1}
    return args

def set_eval(m):
  for name, param in m.named_parameters():
      param.requires_grad = False
  m.eval()
  return m

def get_manifest_dict(args):

    curr_path = Path(__file__).parent.absolute()
    data_path = str(curr_path)+os.sep+'..'+os.sep+'..'+os.sep+args.get('DATA_FOLDER')

    train_manifest = os.path.join(data_path, args.get("TRAIN_MANIFEST"))
    val_manifest = os.path.join(data_path, args.get("VAL_MANIFEST"))
    test_manifest = os.path.join(data_path, args.get("TEST_MANIFEST"))

    return {'train': train_manifest, 'val': val_manifest, 'test':test_manifest}

class ModelCheckpointAtEpochEnd(pl.callbacks.ModelCheckpoint):
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        metrics['epoch'] = trainer.current_epoch
        trainer.checkpoint_callback.on_validation_end(trainer, pl_module)

def finetune(param, args, manifest_dict):
    curr_path = Path(__file__).parent.absolute()
    data_path = str(curr_path)+os.sep+'..'+os.sep+'..'+os.sep+args.get('DATA_FOLDER')
    model_path = str(curr_path)+os.sep+'..'+os.sep+'..'+os.sep+args.get('SAV_CHECKPOINT_PATH')

    config_path = str(curr_path)+os.sep+'..'+os.sep+'..'+os.sep+args.get('CONFIG_PATH')
    
    early_stop_callback = EarlyStopping(
                            monitor=args.get("MONITOR"),
                            patience=args.get('PATIENCE'),
                            mode='min'
                            )

    tag = args.get("EXPERIMENT_TAG")
    language = args.get("LANGUAGE", "unk")
    
    checkpoint_callback = ModelCheckpointAtEpochEnd(
        model_path+os.sep+'jasper_ft_'+language+str(tag)+'_{epoch:02d}',
        monitor=args.get("MONITOR"),
        verbose=True,
        save_top_k=1,
        save_weights_only=False,
        period=1)

    trainer = pl.Trainer(gpus=(args.get('NUMB_GPU')),
                        num_nodes=1, 
                        deterministic=True,
                        distributed_backend = args.get('DISTRIBUTED_BACKEND'),
                        default_root_dir=model_path,
                        max_epochs=args.get('MAX_EPOCHS'),
                        early_stop_callback=early_stop_callback,
                        checkpoint_callback=checkpoint_callback,
                        resume_from_checkpoint=args.get("RESUME_CHECKPOINT_NAME", None))

    if args.get("RESUME_CHECKPOINT_NAME"):
        chkpt = args.get("RESUME_CHECKPOINT_NAME")
        print(f"Resuming from checkpoint {chkpt}")

    param['model']['train_ds']['manifest_filepath'] = manifest_dict['train']
    param['model']['validation_ds']['manifest_filepath'] = manifest_dict['val']

    if args.get('USE_RANDOM_MODEL') is True:
        # To train a randomly initialized model for asr
        print(f"Randomly initializing model {args.get('MODEL_NAME')} for training ASR")
        print('---------------------------------------------------------')
        asr_model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(param['model']), trainer=trainer)

    else:
        if args.get('USE_ENG_ASR'):
            # Initialize model with ENG-ASR
            print(f"Using pretrained model {args.get('MODEL_NAME')} for finetuning ASR")
            print('--------------------------------------------------------------')

            asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(
                                    model_name=args.get('MODEL_NAME'))
        elif args.get("checkpoint_file") is not None:      
            model_full_path = os.path.join(model_path, args.get("checkpoint_file"))
            
            asr_model = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(model_full_path)

        else:
            print("No training configuration was selected. Check the training parameters given")
            print('---------------------------------------------------------')
            return
        
        new_opt = copy.deepcopy(param['model']['optim'])
        cfg = copy.deepcopy(asr_model.cfg)

        # # OmegaConf won't allow you to add new config items, so we temporarily disable this safeguard.
        OmegaConf.set_struct(cfg, False)

        # Now we update the config
        asr_model.setup_optimization(optim_config=DictConfig(new_opt))

        asr_model.set_trainer(trainer)
        asr_model.setup_training_data(train_data_config=param['model']['train_ds'])
        asr_model.setup_validation_data(val_data_config=param['model']['validation_ds'])
    
    # added this to use cer or change vocab 
    asr_model.change_vocabulary = change_vocabulary2
    asr_model.change_vocabulary(asr_model,
        new_vocabulary=param['model']['labels']
        )
        
    if args.get('FREEZE_FEATURE_EXTRACTOR'):
        print('Acoustic Encoder is been used as a feature extractor only')
        print('---------------------------------------------------------')
        asr_model.encoder.apply(set_eval)

    asr_model.summarize()
    
    trainer.fit(asr_model)
    
    print(f'Best checkpoint during training is {checkpoint_callback.best_model_path}')
    
    final_save_filename = os.path.join(model_path, 'jasper_ft_'+str(language)+str(tag)+"_last.ckpt")
    
    trainer.save_checkpoint(final_save_filename)
    
    print('Model finetuning completed, clearing cache')
    
    # clear cache
    if 'asr_model' in locals(): 
        del asr_model
    
    torch.cuda.empty_cache()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-file',
                        help="Checkpoint file .ckpt of phoneme pretrained model to use for finetuning")
    parser.add_argument('--freeze-feature-extractor', dest='FREEZE_FEATURE_EXTRACTOR',
                        default=False, action='store_true', 
                        help="Whether to freexe the backbone or update it during finetuning")
    parser.add_argument('--use-eng-asr', dest='USE_ENG_ASR',
                        default=False, action='store_true', 
                        help="Whether to use Eng-ASR as backbone during finetuning")
    parser.add_argument('--use-random-model', dest='USE_RANDOM_MODEL', 
                        default=False, action='store_true',
                        help="use a pretrained model to initialize the weights")

    args = parser.parse_args()
    print(vars(args))
    args = collate_args(args1=vars(args), 
                        args2=vars(args_default)
                        )
    curr_path = Path(__file__).parent.absolute()
    config_path = str(curr_path)+os.sep+'..'+os.sep+'..'+os.sep+args.get('CONFIG_PATH')
    data_path = str(curr_path)+os.sep+'..'+os.sep+'..'+os.sep+args.get('DATA_FOLDER')
    
    
    manifest_dict = get_manifest_dict(args)

    # Load jasper parameters
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        param = yaml.load(f)
    print('Labels: ', type(param['model']['labels']))

    finetune(param, args, manifest_dict)
    
    print('Model finetuning completed')
    


