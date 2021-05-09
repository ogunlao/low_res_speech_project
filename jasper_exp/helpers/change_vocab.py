from typing import List
import copy
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer import WER
from omegaconf import OmegaConf
from nemo.utils import logging

    
def change_vocabulary2(self, new_vocabulary: List[str]):
    """
    Changes vocabulary used during CTC decoding process. Use this method when fine-tuning on from pre-trained model.
    This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
    use it if you want to use pretrained encoder when fine-tuning on a data in another language, or when you'd need
    model to learn capitalization, punctuation and/or special characters.
    If new_vocabulary == self.decoder.vocabulary then nothing will be changed.
    Args:
        new_vocabulary: list with new vocabulary. Must contain at least 2 elements. Typically, \
        this is target alphabet.
    Returns: None
    """
    # if self.decoder.vocabulary == new_vocabulary:
    #     logging.warning(f"Old {self.decoder.vocabulary} and new {new_vocabulary} match. Not changing anything.")
    
    if new_vocabulary is None or len(new_vocabulary) == 0:
        raise ValueError(f'New vocabulary must be non-empty list of chars. But I got: {new_vocabulary}')
    decoder_config = self.decoder.to_config_dict()
    new_decoder_config = copy.deepcopy(decoder_config)
    if 'vocabulary' in new_decoder_config:
        new_decoder_config['vocabulary'] = new_vocabulary
        new_decoder_config['num_classes'] = len(new_vocabulary)
    else:
        new_decoder_config['params']['vocabulary'] = new_vocabulary
        new_decoder_config['params']['num_classes'] = len(new_vocabulary)
    del self.decoder
    self.decoder = EncDecCTCModel.from_config_dict(new_decoder_config)
    del self.loss
    self.loss = CTCLoss(num_classes=self.decoder.num_classes_with_blank - 1, zero_infinity=True)
    self._wer = WER(vocabulary=self.decoder.vocabulary, batch_dim_index=0, use_cer=True, ctc_decode=True)

    # Update config
    OmegaConf.set_struct(self._cfg.decoder, False)
    self._cfg.decoder = new_decoder_config
    OmegaConf.set_struct(self._cfg.decoder, True)
