import os
from nemo.collections.asr.metrics.wer import WER
from pathlib import Path

def eval_model(asr_model, params, args, manifest_file='', 
                valid=True, test=False, train=False):

    # Bigger batch-size = bigger throughput
    params['model']['validation_ds']['batch_size'] = int(args.get('EVALUATION_BS'))
    curr_path = curr_path = Path(__file__).parent.absolute()
    data_path = str(curr_path)+os.sep+'..'+os.sep+'..'+os.sep+args.get('DATA_FOLDER')

    if valid:
        print('Evaluation on the validation...')
        asr_model.setup_test_data(test_data_config=params['model']['validation_ds'])

    elif test:
        print('Evaluation on the test...')
        test_manifest = os.path.join(data_path, manifest_file)
        params['model']['validation_ds']['manifest_filepath'] = test_manifest
        asr_model.setup_test_data(test_data_config=params['model']['validation_ds'])

    elif train:
        print('Evaluation on the train...')
        train_manifest = params['model']['train_ds']['manifest_filepath']
        params['model']['validation_ds']['manifest_filepath'] = train_manifest
        asr_model.setup_test_data(test_data_config=params['model']['validation_ds'])

    asr_model.cuda()
    asr_model.eval()

    # We will be computing Character Error Rate (CER) metric between our hypothesis and predictions.
    # CER is computed as numerator/denominator.
    # We'll gather all the test batches' numerators and denominators.
    cer_nums = []
    cer_denoms = []

    # Loop over all test batches.
    # Iterating over the model's `test_dataloader` will give us:
    # (audio_signal, audio_signal_length, transcript_tokens, transcript_length)
    # See the AudioToCharDataset for more details.

    cer = WER(vocabulary=asr_model.decoder.vocabulary,
            use_cer=True,)
    for test_batch in asr_model.test_dataloader():
        test_batch = [x.cuda() for x in test_batch]
        targets = test_batch[2]
        targets_lengths = test_batch[3]        
        log_probs, encoded_len, greedy_predictions = asr_model(
            input_signal=test_batch[0], input_signal_length=test_batch[1],
        )
        
        cer_num, cer_denom = cer(greedy_predictions, targets, targets_lengths)
        cer_nums.append(cer_num.detach().cpu().numpy())
        cer_denoms.append(cer_denom.detach().cpu().numpy())

    if valid:
        print(f"Val PER = {sum(cer_nums)/sum(cer_denoms)}")
    elif train:
        print(f"Train PER = {sum(cer_nums)/sum(cer_denoms)}")
    else:
        print(f"Test PER = {sum(cer_nums)/sum(cer_denoms)}")

    print('Evaluation completed')
    print('---------------------------------------------')