import torch
import torch.nn.functional as F
import shutil
import time
import datetime
import numpy as np
import progressbar
from multiprocessing import Pool

try:
  from utils import args_ctc as args
  from utils import make_dirs
except:
  from .utils import args_ctc as args
  from .utils import make_dirs
import wget
import sys

from cpc.eval.common_voices_eval import SingleSequenceDataset, parseSeqLabels, findAllSeqs
from cpc.feature_loader import loadModel
try:
  from cpc_models import CharacterClassifier
except:
  from .cpc_models import CharacterClassifier

try:
  from cpc_eval import get_cer
except:
  from .cpc_eval import get_cer
import pytorch_warmup as warmup

import random
import numpy as np
import pandas as pd

try:
  from preprocess import get_pseudolabels
except:
  from preprocess import get_pseudolabels

device = torch.device("cuda:0" if args.DEVICE else "cpu")

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def create_manifest(df, file_path):
    with open(file_path, "w+") as f:
        for i in range(len(df)):
            wav_name = df.iloc[i]['path']
            wav_name = wav_name[:-4]
            print(wav_name, f)
        

def train_one_epoch_ctc(cpc_model, 
                        character_classifier, 
                        loss_criterion, 
                        data_loader, 
                        optimizer,
                        lr_sch, warmup_scheduler, 
                        epoch):
  
  cpc_model.train()
  character_classifier.train()
  loss_criterion.train()
  
  avg_loss = 0
  avg_accuracy = 0
  n_items = 0
  for step, full_data in enumerate(data_loader):

    x, x_len, y, y_len = full_data

    x_batch_len = x.shape[-1]
    x, y = x.to(device), y.to(device)

    bs=x.size(0)
    optimizer.zero_grad()
    context_out, enc_out, _ = cpc_model(x.to(device), y.to(device))
  
    scores = character_classifier(context_out)
    scores = scores.permute(1, 0, 2)
    scores = F.log_softmax(scores, 2)
    yhat_len = torch.tensor([int(scores.shape[0]*x_len[i]/(x_batch_len)) for i in range(scores.shape[1])]) # this is an approximation, should be good enough
    #print(yhat_len)
    loss = loss_criterion(scores, y.to(device), yhat_len, y_len)
    loss.backward()
    optimizer.step()
    lr_sch.step(lr_sch.last_epoch+1)
    warmup_scheduler.dampen()
    # print(optimizer.param_groups[0]['lr'])
    
    avg_loss+=loss.item()*bs
    n_items+=bs
  avg_loss/=n_items
  return avg_loss

def validation_step(cpc_model, 
                    character_classifier, 
                    loss_criterion, 
                    data_loader, args=args):

  cpc_model.eval()
  character_classifier.eval()
  avg_loss = 0
  avg_accuracy = 0
  n_items = 0
  
  if args.VAL_DF:
    val_df = pd.read_csv(args.VAL_DF)
  with torch.no_grad():
    for step, full_data in enumerate(data_loader):
      
      if args.VAL_DF and step == 0:        
          get_pseudolabels(val_df, data_loader, cpc_model, character_classifier, args)
        
      x, x_len, y, y_len = full_data

      x_batch_len = x.shape[-1]
      x, y = x.to(device), y.to(device)

      bs=x.size(0)
      context_out, enc_out, _ = cpc_model(x.to(device),y.to(device))
    
      scores = character_classifier(context_out)
      scores = scores.permute(1, 0, 2)
      scores = F.log_softmax(scores, 2)
      yhat_len = torch.tensor([int(scores.shape[0]*x_len[i]/x_batch_len) for i in range(scores.shape[1])]) # this is an approximation, should be good enough
      loss = loss_criterion(scores,y.to(device),yhat_len,y_len)
      avg_loss+=loss.item()*bs
      n_items+=bs
  avg_loss/=n_items
  return avg_loss


def run_ctc(cpc_model, character_classifier, 
            loss_criterion, data_loader_train, 
            data_loader_val, optimizer,
            lr_sch, warmup_scheduler, n_epoch,
            patience=5, args=args):
  
  t0 = time.time()
  losses_train = []
  losses_val = []
  min_loss = float("inf")
  best_epoch = 0
  patience_count = 0

  try:
      for epoch in range(n_epoch):

        print(f"Running epoch {epoch+1} / {n_epoch}")
        
        loss_train = train_one_epoch_ctc(cpc_model, character_classifier, 
                                         loss_criterion, data_loader_train, 
                                         optimizer, lr_sch, warmup_scheduler,
                                         epoch)
        losses_train.append(loss_train)
        print("-------------------")
        print(f"Training dataset :")
        print(f"Average loss : {loss_train}.")

        print("-------------------")
        print("Validation dataset")
        loss_val = validation_step(cpc_model, character_classifier, loss_criterion, data_loader_val)
        losses_val.append(loss_val)
        # lr_sch.step(epoch)
        # warmup_scheduler.dampen()
        # print(optimizer.param_groups[0]['lr'])

        if loss_val < min_loss:
          min_loss = loss_val
          save_checkpoint(cpc_model, character_classifier)
          patience_count = 0
          best_epoch = epoch
        else:
          patience_count += 1
          print(f"No model improvement at epoch")

        print(f"Average loss : {loss_val}")
        print("-------------------")
        print()
        if patience_count > patience:
          print(f"No model improvement at epoch {epoch}, Aborting")
          break
      print(f"loading best model at epoch {best_epoch} with loss {min_loss}")
      cpc_model, character_classifier = load_checkpoint(cpc_model, character_classifier)
      
      # save a copy in drive
      # save_final_checkpoint(cpc_model, character_classifier)

      print(f'Training finished in {(time.time() - t0)/3600} hours')
  except KeyboardInterrupt:
      print("Loading checkpoint after interrupt...")
      cpc_model, character_classifier = load_checkpoint(cpc_model, character_classifier)
      # save a copy in drive
      # save_final_checkpoint(cpc_model, character_classifier)

  return losses_train, losses_val, cpc_model, character_classifier #, cer_val


def save_checkpoint(model, classifier, path=args.CHECKPOINT_SAVE_PATH):
  torch.save(model.state_dict(), path)
  torch.save(classifier.state_dict(), path+".classifier")


def load_checkpoint(model, classifier, path=args.CHECKPOINT_SAVE_PATH):
  model.load_state_dict(torch.load(path))
  classifier.load_state_dict(torch.load(path+".classifier"))

  return model, classifier


def save_final_checkpoint(path=args.CHECKPOINT_SAVE_PATH, args=args):
  make_dirs(args.FINAL_MODEL_SAVE_PATH)

  pid=os.getpid()
  dt=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

  save_model_as = 'cpc_checkpoint_ps_'+str(dt)+str(pid)+'.ckpt'
  shutil.copy2(args.CHECKPOINT_SAVE_PATH, os.path.join(args.FINAL_MODEL_SAVE_PATH, save_model_as))
  shutil.copy2(args.CHECKPOINT_SAVE_PATH+".classifier", os.path.join(args.FINAL_MODEL_SAVE_PATH, save_model_as+".classifier"))

def download_ckpt(ckpt_path):
    make_dirs(ckpt_path)
    print(f"Downloadin checkpoint data to {ckpt_path}")
    wget.download(url="https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/not_hub/2levels_6k_top_ctc/checkpoint_30.pt",
         out=ckpt_path)
    wget.download(url="https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/not_hub/2levels_6k_top_ctc/checkpoint_logs.json",
         out=ckpt_path)
    wget.download(url="https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/not_hub/2levels_6k_top_ctc/checkpoint_args.json",
         out=ckpt_path)    

def create_dataloader(train_data_path, val_data_path, test_data_path, args=args):
    # Load data loader
    letters_labels, N_LETTERS = parseSeqLabels(args.PATH_LETTER_DATA_CER)
    
    data_train_cer, _ = findAllSeqs(train_data_path, extension=args.DATA_EXT)
    dataset_train_non_aligned = SingleSequenceDataset(train_data_path, data_train_cer, letters_labels)

    data_val_cer, _ = findAllSeqs(val_data_path, extension=args.DATA_EXT)
    dataset_val_non_aligned = SingleSequenceDataset(val_data_path, data_val_cer, letters_labels)

    data_test_cer, _ = findAllSeqs(test_data_path, extension=args.DATA_EXT)
    dataset_test_non_aligned = SingleSequenceDataset(test_data_path, data_test_cer, letters_labels)
    
    data_loader_train_letters = torch.utils.data.DataLoader(dataset_train_non_aligned, batch_size=args.TRAIN_BATCH_SIZE,
                                                    shuffle=True,)
    data_loader_val_letters = torch.utils.data.DataLoader(dataset_val_non_aligned, batch_size=args.VAL_BATCH_SIZE,
                                                shuffle=False,)
    data_loader_test_letters = torch.utils.data.DataLoader(dataset_test_non_aligned, batch_size=args.VAL_BATCH_SIZE,
                                                shuffle=False,)
    
    return {'train': data_loader_train_letters,
            'val': data_loader_val_letters,
            'test': data_loader_test_letters}

def create_model(args):
  
  import torch
  import argparse

  checkpoint_url = 'https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/60k_epoch4-d0f474de.pt'
  checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url,progress=False, map_location='cpu')

  from cpc.model import CPCModel as cpcmodel
  from cpc.cpc_default_config import get_default_cpc_config
  from cpc.feature_loader import getEncoder, getAR, loadArgs
  locArgs = get_default_cpc_config()

  loadArgs(locArgs, argparse.Namespace(**checkpoint["config"]))
  encoderNet = getEncoder(locArgs)
  arNet_context = getAR(locArgs)
  cpc_model = cpcmodel(encoderNet, arNet_context)

  cpc_model.load_state_dict(checkpoint['weights'])
  cpc_model = cpc_model.cuda()
  HIDDEN_CONTEXT_MODEL = 256
  character_classifier = CharacterClassifier(HIDDEN_CONTEXT_MODEL, args.N_LETTERS).to(device)
    
  return cpc_model, character_classifier

def finetune_ckpt(train_data_path, val_data_path, dataloaders, args=args):
    # download_ckpt(ckpt_path="checkpoint_data")
    
    
    letters_labels, N_LETTERS = parseSeqLabels(args.PATH_LETTER_DATA_CER)
    args.N_LETTERS=N_LETTERS # +1 for the blank token
    
    # Load model
    # cpc_model, HIDDEN_CONTEXT_MODEL, HIDDEN_ENCODER_MODEL = loadModel([args.CHECKPOINT_PATH])
    # cpc_model = cpc_model.to(device)
    # character_classifier = CharacterClassifier(HIDDEN_CONTEXT_MODEL, args.N_LETTERS).to(device)
    
    cpc_model, character_classifier = create_model(args)
    if args.FREEZE_ENCODER:
      parameters = character_classifier.parameters()
    else:
      parameters = list(character_classifier.parameters()) + list(cpc_model.parameters())
    
    optim = args.OPTIMIZER
    optimizer = optim(parameters, lr=args.LEARNING_RATE)#, weight_decay=args.WEIGHT_DECAY,)
    
    sched = args.SCHEDULER
    T_max = args.N_EPOCH*len(dataloaders['train'])
    lr_sch = sched(optimizer, T_max=T_max, eta_min=args.MIN_LR)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=args.WARMUP_PERIOD)
    warmup_scheduler.last_step = -1 # initialize the step counter

    loss_ctc = torch.nn.CTCLoss()
    
    data_loader_train_letters = dataloaders['train']
    data_loader_val_letters = dataloaders['val']
    
    losses_train, losses_val, cpc_model, character_classifier = run_ctc(
        cpc_model,
        character_classifier,
        loss_ctc,
        data_loader_train_letters,
        data_loader_val_letters,
        optimizer, lr_sch, warmup_scheduler, n_epoch=args.N_EPOCH, 
        patience=args.PATIENCE)
    
    return cpc_model, character_classifier

if __name__ == "__main__":
    if len(sys.argv) > 2:
        args.PATH_TRAIN_DATA_CER = sys.argv[1]
        args.PATH_VAL_DATA_CER = sys.argv[2]
        args.PATH_TEST_DATA_CER = sys.argv[3]
        args.VAL_DF = sys.argv[4]
        
    
    set_seed(args.SEED)
    
    dataloaders = create_dataloader(args.PATH_TRAIN_DATA_CER, 
                                    args.PATH_VAL_DATA_CER, 
                                    args.PATH_TEST_DATA_CER,)
    
    cpc_model, character_classifier = finetune_ckpt(args.PATH_TRAIN_DATA_CER, args.PATH_VAL_DATA_CER, dataloaders)
    
    val_cer = get_cer(dataloaders["val"], cpc_model, character_classifier)
    test_cer = get_cer(dataloaders["test"], cpc_model, character_classifier, args)
    
    print(f"val cer: {val_cer}, test cer: {test_cer}")
    
    # save to file
    file_path = "result.txt"
    with open(file_path, "a+") as f:
          print(f"val cer: {val_cer}, test cer: {test_cer}", file=f)