import numpy as np
import progressbar
from multiprocessing import Pool
from ctcdecode import CTCBeamDecoder
try:
  from utils import args_ctc as args
except:
  from .utils import args_ctc as args
import time
import torch

device = torch.device("cuda:0" if args.DEVICE else "cpu")

def cut_data(seq, sizeSeq):
    maxSeq = sizeSeq.max()
    return seq[:, :maxSeq]


def prepare_data(data):
    seq, sizeSeq, char, sizeChar = data
    seq = seq.cuda()
    char = char.cuda()
    sizeSeq = sizeSeq.cuda().view(-1)
    sizeChar = sizeChar.cuda().view(-1)

    seq = cut_data(seq.permute(0, 2, 1), sizeSeq).permute(0, 2, 1)
    return seq, sizeSeq, char, sizeChar
  

def get_CER_sequence(ref_seq, target_seq):
  n = len(ref_seq)
  m = len(target_seq)

  D = np.zeros((n+1, m+1))
  for i in range(1, n+1):
    D[i,0] = D[i-1, 0]+1
  for j in range(1, m+1):
    D[0,j] = D[0, j-1]+1
  
  # compute the alignment

  for i in range(1, n+1):
    for j in range(1, m+1):
      D[i,j] = min(
          D[i-1, j]+1,
          D[i-1, j-1]+1,
          D[i, j-1]+1,
          D[i-1, j-1]+ 0 if ref_seq[i-1]==target_seq[j-1] else float("inf")
      )
  return D[n,m]/len(ref_seq)


def get_cer(dataloader,
            cpc_model,
            character_classifier, args=args):
  t0 = time.time()

  downsampling_factor = 1
  cpc_model.eval()
  character_classifier.eval()

  avgCER = 0
  nItems = 0

  print("Starting the CER computation through beam search")
  bar = progressbar.ProgressBar(maxval=len(dataloader))
  bar.start()

  decoder = CTCBeamDecoder(args.CHARS, log_probs_input=False, blank_id=0, beam_width=args.BEAM_WIDTH, cutoff_top_n=args.CUT_OFF_TOP_N) #model_path="/content/wiki_00.lm.arpa")
  
  for index, data in enumerate(dataloader):

    bar.update(index)
    with torch.no_grad():
        seq, sizeSeq, phone, sizePhone = prepare_data(data)
        x_batch_len = seq.shape[-1]
        c_feature, _, _ = cpc_model(seq.to(device),phone.to(device))
        bs = c_feature.size(0)
        sizeSeq = sizeSeq / downsampling_factor
        predictions = torch.nn.functional.softmax(
            character_classifier(c_feature), dim=2
            ).cpu()
        phone = phone.cpu()
        sizeSeq = sizeSeq.cpu()
        sizePhone = sizePhone.cpu()
        # print("predictions",predictions.argmax(2)[0])
        # print(phone[0])
        
        seq_len = torch.tensor([int(predictions.shape[1]*sizeSeq[i]/(x_batch_len)) for i in range(predictions.shape[0])]) # this is an approximation, should be good enough
        #print(seq_len)
        
        output, scores, timesteps, out_seq_len = decoder.decode(predictions, seq_lens=seq_len)
        
        output=output[torch.arange(bs), scores.argmax(1),:]
        out_seq_len= out_seq_len[torch.arange(bs),scores.argmax(1)]
        data_cer = []
        for b in range(bs):
          # print(sizePhone[b], out_seq_len[b])
          data_cer.append((phone[b][:sizePhone[b].item()], output[b][:out_seq_len[b].item()]))
          
        #data_cer = [(predictions[b].argmax(1),  phone[b]) for b in range(bs)]
        # data_cer = [(predictions[b], sizeSeq[b], phone[b], sizePhone[b],
        #               "criterion.module.BLANK_LABEL") for b in range(bs)]
        with Pool(bs) as p:
            poolData = p.starmap(get_CER_sequence, data_cer)
        avgCER += sum([x for x in poolData])
        nItems += len(poolData)

  bar.finish()

  avgCER /= nItems

  print(f"Average CER {avgCER}")
  return avgCER