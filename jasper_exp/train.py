# --- Config Information ---#
import nemo
import nemo.collections.asr as nemo_asr

from ruamel.yaml import YAML
from args_nemo import args as args_default
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

from eval_model import eval_model

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
        patience=args.get('PATIENCE'),
    )

    checkpoint_callback = ModelCheckpointAtEpochEnd(
        model_path+os.sep+'jasper_ph_rw_'+'_{epoch:02d}',
        verbose=True,
        save_top_k=1,
        save_weights_only=False,
        period=1)

    trainer = pl.Trainer(gpus=args.get('NUMB_GPU'),
                        num_nodes=args.get('NUMB_GPU'), 
                        distributed_backend='ddp',
                        max_epochs=args.get('MAX_EPOCHS'),
                        early_stop_callback=early_stop_callback,
                        checkpoint_callback=checkpoint_callback)

    train_manifest = os.path.join(data_path, 'train_manifest.json')
    val_manifest = os.path.join(data_path, 'val_manifest.json')

    params['model']['train_ds']['manifest_filepath'] = train_manifest
    params['model']['validation_ds']['manifest_filepath'] = val_manifest

    if args.get('USE_PRETRAINED') is False:
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
        asr_model.change_vocabulary(
                new_vocabulary=params['model']['labels']
                )

    asr_model.summarize()
    trainer.fit(asr_model)

    asr_model.save_to(os.path.join(model_path, 'asr_model_jasper_ph.nemo'))
    return asr_model

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
    
    try:
        asr_model = train(params, args)

        #Evaluate on the training
        eval_model(asr_model, params, args, 
                    valid=False, train=True)

        #Evaluate on the validation
        eval_model(asr_model, params, args)
    finally:
        # clear cache
        del asr_model
        # torch.cuda.empty_cache()
