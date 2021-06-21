# --- Config Information ---#
import nemo
import nemo.collections.asr as nemo_asr

from ruamel.yaml import YAML
from configs.args_nemo_train import args as args_default
from omegaconf import DictConfig
from omegaconf import OmegaConf
import argparse

import os, datetime
from pathlib import Path
import sys
import copy

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from helpers.change_vocab import change_vocabulary2

def collate_args(args1, args2):
    args = {**args2, **args1}
    return args

class ModelCheckpointAtEpochEnd(pl.callbacks.ModelCheckpoint):
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        metrics['epoch'] = trainer.current_epoch
        trainer.checkpoint_callback.on_validation_end(trainer, pl_module)

def train(params, args):
    curr_path = Path(__file__).parent.absolute()
    data_path = str(curr_path)+os.sep+'..'+os.sep+'..'+os.sep+args.get('DATA_FOLDER')
    model_path = str(curr_path)+os.sep+'..'+os.sep+'..'+os.sep+args.get('SAV_CHECKPOINT_PATH')
    config_path = str(curr_path)+os.sep+'..'+os.sep+'..'+os.sep+args.get('CONFIG_PATH')

    early_stop_callback = EarlyStopping(
                            monitor=args.get("MONITOR"),
                            patience=args.get('PATIENCE'),
                            mode='min'
                            )

    tag = args.get("EXPERIMENT_TAG", "01")
    language = args.get("LANGUAGE", "unk")
    checkpoint_callback = ModelCheckpointAtEpochEnd(
        model_path+os.sep+'jasper_ph_'+language+str(tag)+'_{epoch:02d}',
        monitor=args.get("MONITOR"),
        verbose=True,
        save_top_k=1,
        save_weights_only=False,
        period=1)

    trainer = pl.Trainer(gpus=(args.get('NUMB_GPU')),
                        num_nodes=1,
                        distributed_backend = args.get('DISTRIBUTED_BACKEND'), 
                        max_epochs=args.get('MAX_EPOCHS'),
                        default_root_dir=model_path,
                        checkpoint_callback=checkpoint_callback,
                        early_stop_callback=early_stop_callback,
                        resume_from_checkpoint=args.get("RESUME_CHECKPOINT_NAME", None))
    
    if args.get("RESUME_CHECKPOINT_NAME"):
        chkpt = args.get("RESUME_CHECKPOINT_NAME")
        print(f"Will resuming from checkpoint {chkpt}")
        
    train_manifest = os.path.join(data_path, args.get("TRAIN_MANIFEST"))
    val_manifest = os.path.join(data_path, args.get("VAL_MANIFEST"))

    params['model']['train_ds']['manifest_filepath'] = train_manifest
    params['model']['validation_ds']['manifest_filepath'] = val_manifest

    if args.get('USE_PRETRAINED') is False:
        print("Random model will be initialized for training")
        asr_model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']), 
                                                trainer=trainer)
    else:
        print(f"Using pretrained model {args.get('MODEL_NAME')} for training")
        print('--------------------------------------------------------------')
        asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(
                                            model_name=args.get('MODEL_NAME'))

        new_opt = copy.deepcopy(params['model']['optim'])
        cfg = copy.deepcopy(asr_model.cfg)

        # # OmegaConf won't allow you to add new config items, so we temporarily disable this safeguard.
        OmegaConf.set_struct(cfg, False)

        # Now we update the config
        asr_model.setup_optimization(optim_config=DictConfig(new_opt))

        asr_model.set_trainer(trainer)
        asr_model.setup_training_data(train_data_config=params['model']['train_ds'])
        asr_model.setup_validation_data(val_data_config=params['model']['validation_ds'])
    
    # setting this to enable use_cer = True
    asr_model.change_vocabulary = change_vocabulary2
    asr_model.change_vocabulary(asr_model,
            new_vocabulary=params['model']['labels']
            )

    asr_model.summarize()
    
    trainer.fit(asr_model)
    
    print(f'Best checkpoint during training is {checkpoint_callback.best_model_path}')
    
    final_save_filename = os.path.join(model_path, 'jasper_ph_'+str(language)+str(tag)+"_last.ckpt")
    trainer.save_checkpoint(final_save_filename)
    
    # clear cache
    if 'asr_model' in locals(): 
        del asr_model
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_pretrained', dest='USE_PRETRAINED', 
                        default=False, action='store_true',
                        help="use a pretrained model to initialize the weights")

    args = parser.parse_args()
    
    args = collate_args(args1=vars(args), 
                        args2=vars(args_default)
                        )

    curr_path = Path(__file__).parent.absolute()
    config_path = str(curr_path)+os.sep+'..'+os.sep+'..'+os.sep+args.get('CONFIG_PATH')

    # Load jasper parameters
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)
    
    train(params, args)
    
    print('Model training completed')