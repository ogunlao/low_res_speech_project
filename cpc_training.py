import torch.nn.functional as F
import shutil
import time
import datetime
import numpy as np
import progressbar
from multiprocessing import Pool

def create_manifest(df, file_path):
    with open(file_path, "w+") as f:
        for i in range(len(df)):
            wav_name = df.iloc[i]['path']
            wav_name = wav_name[:-4]
            print(wav_name, file=f)
        

def train_one_epoch_ctc(cpc_model, 
                        character_classifier, 
                        loss_criterion, 
                        data_loader, 
                        optimizer):
  
  cpc_model.train()
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
    avg_loss+=loss.item()*bs
    n_items+=bs
  avg_loss/=n_items
  return avg_loss

def validation_step(cpc_model, 
                    character_classifier, 
                    loss_criterion, 
                    data_loader):

  cpc_model.eval()
  character_classifier.eval()
  avg_loss = 0
  avg_accuracy = 0
  n_items = 0
  with torch.no_grad():
    for step, full_data in enumerate(data_loader):

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
            lr_sch, n_epoch,
            patience=5, args=args):
  
  t0 = time.time()
  losses_train = []
  losses_val = []
  min_loss = float("inf")
  best_epoch = 0
  patience_count = 0

  try:
      for epoch in range(n_epoch):

        print(f"Running epoch {epoch + 1} / {n_epoch}")
        loss_train = train_one_epoch_ctc(cpc_model, character_classifier, loss_criterion, data_loader_train, optimizer)
        losses_train.append(loss_train)
        print("-------------------")
        print(f"Training dataset :")
        print(f"Average loss : {loss_train}.")

        print("-------------------")
        print("Validation dataset")
        loss_val = validation_step(cpc_model, character_classifier, loss_criterion, data_loader_val)
        losses_val.append(loss_val)
        lr_sch.step()
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
      save_final_checkpoint(cpc_model, character_classifier)

      print(f'Training finished in {(time.time() - t0)/3600} hours')
  except KeyboardInterrupt:
      print("Loading checkpoint after interrupt...")
      cpc_model, character_classifier = load_checkpoint(cpc_model, character_classifier)
      # save a copy in drive
      save_final_checkpoint(cpc_model, character_classifier)

  return losses_train, losses_val, cpc_model, character_classifier #, cer_val


def save_checkpoint(model, classifier, path=args.CHECKPOINT_SAVE_PATH):
  torch.save(model.state_dict(), path)
  torch.save(classifier.state_dict(), path+".classifier")


def load_checkpoint(model, classifier, path=args.CHECKPOINT_SAVE_PATH):
  model.load_state_dict(torch.load(path))
  classifier.load_state_dict(torch.load(path+".classifier"))

  return model, classifier


def save_final_checkpoint(model, classifier, path=args.CHECKPOINT_SAVE_PATH, args=args):
  make_dirs(args.FINAL_MODEL_SAVE_PATH)

  pid=os.getpid()
  dt=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

  save_model_as = 'cpc_checkpoint_ps_'+str(dt)+str(pid)+'.ckpt'
  shutil.move(args.CHECKPOINT_SAVE_PATH, os.path.join(args.FINAL_MODEL_SAVE_PATH, save_model_as))
  shutil.move(args.CHECKPOINT_SAVE_PATH+".classifier", os.path.join(args.FINAL_MODEL_SAVE_PATH, save_model_as+".classifier"))

