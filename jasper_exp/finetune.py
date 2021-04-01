import nemo
import nemo.collections.asr as nemo_asr

from ruamel.yaml import YAML

from pathlib import Path
import os

from omegaconf import DictConfig
from omegaconf import OmegaConf
import argparse
import pandas as pd

from args_nemo import args as args_default
from build_manifest import build_manifest
from eval_model import eval_model

import sys
import copy

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


def collate_args(args1, args2):
    args = {**args2, **args1}
    return args

def set_eval(m):
  for name, param in m.named_parameters():
      param.requires_grad = False
  m.eval()
  return m

class ModelCheckpointAtEpochEnd(pl.callbacks.ModelCheckpoint):
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        metrics['epoch'] = trainer.current_epoch
        trainer.checkpoint_callback.on_validation_end(trainer, pl_module)


# Restore the model

# asr_model = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(ckpt_path)
# asr_model = nemo_asr.models.EncDecCTCModel.restore_from(MODEL_PATH)

def build_manifest_file(args):

    curr_path = Path(__file__).parent.absolute()
    data_path = str(curr_path)+os.sep+'..'+os.sep+'..'+os.sep+args.get('DATA_FOLDER')

    raw_audio_paths = args.get('RAW_AUDIO_PATH')
    sampled_data_dir = os.path.join(data_path, args.get('SAMPLED_DATA_FOLDER'))

    train_ft_df = pd.read_csv(os.path.join(data_path, args.get('FINETUNE_CSV')))

    val_ft_df = pd.read_csv(os.path.join(data_path, raw_audio_paths, '..', args.get('VAL_FT_CSV')), 
                            sep='\t')
    test_ft_df = pd.read_csv(Path(os.path.join(data_path, raw_audio_paths, '..', args.get('TEST_FT_CSV'))),
                            sep='\t')

    clips = {}
    with open(data_path+os.sep+args.get('DURATION_SAV_FILE'), 'r') as f:
        for line in f:
            file_name, duration = line.split()
            clips[file_name] = duration
    
    val_ft_df['path'] = val_ft_df['path'].apply(
        lambda x: x[:-4]+'.wav')

    val_ft_df['duration'] = val_ft_df['path'].apply(
        lambda x: clips.get(x, 0.0))

    test_ft_df['path'] = test_ft_df['path'].apply(
        lambda x: x[:-4]+'.wav')

    test_ft_df['duration'] = test_ft_df['path'].apply(
        lambda x: clips.get(x, 0.0))

    columns_to_select = ['path', 'sentence', 'gender', 'duration']
    val_ft_df = val_ft_df[columns_to_select]
    test_ft_df = test_ft_df[columns_to_select]


    # Building Manifests
    print("******")

    train_manifest = os.path.join(data_path, 'train_manifest_ft.json')
    val_manifest = os.path.join(data_path, 'val_manifest_ft.json')
    test_manifest = os.path.join(data_path, 'test_manifest_ft.json')
    
    # build_manifest(train_ft_df,  # TODO: Uncomment
    #             train_manifest,
    #             sampled_data_dir,
    #             )
    
    # build_manifest(val_ft_df, 
    #             val_manifest,
    #             sampled_data_dir,
    #             )  
    
    # build_manifest(test_ft_df, 
    #             test_manifest,
    #             sampled_data_dir,
    #             )

    return {'train': train_manifest, 'val': val_manifest, 'test':test_manifest}

    print("Training, validation and test manifests created.")
    print('---------------------------------------------------------')
    print("***Done***")

def finetune(params, args, manifest_dict):
    curr_path = Path(__file__).parent.absolute()
    data_path = str(curr_path)+os.sep+'..'+os.sep+'..'+os.sep+args.get('DATA_FOLDER')
    model_path = str(curr_path)+os.sep+'..'+os.sep+'..'+os.sep+args.get('SAV_CHECKPOINT_PATH')
    config_path = str(curr_path)+os.sep+'..'+os.sep+'..'+os.sep+args.get('CONFIG_PATH_FT')

    early_stop_callback = EarlyStopping(
        patience=args.get('PATIENCE'),
    )

    checkpoint_callback = ModelCheckpointAtEpochEnd(
        model_path+os.sep+'jasper_ft_rw_'+'_{epoch:02d}',
        verbose=True,
        save_top_k=1,
        save_weights_only=False,
        period=1)

    trainer = pl.Trainer(gpus=[args.get('NUMB_GPU')], 
                        max_epochs=args.get('MAX_EPOCHS'),
                        early_stop_callback=early_stop_callback,
                        checkpoint_callback=checkpoint_callback)

    # train_manifest = os.path.join(data_path, 'train_manifest_ft.json')
    # val_manifest = os.path.join(data_path, 'val_manifest_ft.json')

    params['model']['train_ds']['manifest_filepath'] = manifest_dict['train']
    params['model']['validation_ds']['manifest_filepath'] = manifest_dict['val']

    if args.get('USE_RANDOM_MODEL') is True:
        # To train a randomly initialized model for asr
        print(f"Randomly initializing model {args.get('MODEL_NAME')} for training ASR")
        print('---------------------------------------------------------')
        asr_model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']), trainer=trainer)

    else:
        # asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(
        #                                     model_name=args.get('MODEL_NAME'))
        if args.get('USE_ENG_ASR'):
            # Initialize model with ENG-ASR
            print(f"Using pretrained model {args.get('MODEL_NAME')} for finetuning ASR")
            print('--------------------------------------------------------------')

            asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(
                                    model_name=args.get('MODEL_NAME'))
        elif args.get("checkpoint_file") is not None:      
            model_full_path = os.path.join(model_path, args.get("checkpoint_file"))

            # Initialize model with phoneme classifier                  
            asr_model = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(model_full_path)
        else:
            print("No training configuration was selected. Check the training parameters given")
            print('---------------------------------------------------------')
            return

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
               new_vocabulary=
                params['model']['labels']
               )

        if args.get('FREEZE_FEATURE_EXTRACTOR'):
            print('Acoustic Encoder is been used as a feature extractor only')
            print('---------------------------------------------------------')
            asr_model.encoder.apply(set_eval)

    asr_model.summarize()
    trainer.fit(asr_model)

    asr_model.save_to(os.path.join(model_path, 'asr_model_jasper_ft.nemo'))
    return asr_model

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
    config_path = str(curr_path)+os.sep+'..'+os.sep+'..'+os.sep+args.get('CONFIG_PATH_FT')

    manifest_dict = build_manifest_file(args)

    # Load jasper parameters
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)

    print(params['model']['labels'])
    try:
        asr_model = finetune(params, args, manifest_dict)

        if asr_model is not None:
            #Evaluate on the validation
            eval_model(asr_model, params, args, valid=True)

            #Evaluate on the test
            eval_model(asr_model, params, args,  
                        manifest_file=manifest_dict['test'],
                        valid=False, test=True,)
    finally:
        del asr_model
        # torch.cuda.empty_cache()
    


