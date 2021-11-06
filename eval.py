#!/usr/bin/env python3
from module.data_prep import data_prep
from module.processor import Wav2Vec2Processor
from module.model import Wav2Vec2ForCTC

import torch
import torch.nn.functional as F
from module.trainer import DataCollatorCTCWithPadding, BatchRandomSampler
from module.decoder import KenLMDecoder
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
import argparse
import logging
import sys
import os

wer_metric = datasets.load_metric("wer")
cer_metric = datasets.load_metric("cer")


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


def get_data(
    processor,
    dataset,
    name,
    split,
    data_path,
    batch_size=1):

    dataset = data_prep(processor, dataset, name, split, batch_size, data_path)
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        collate_fn=data_collator
    )

def write_result(
    data,
    file_path):
    with open(file_path, "a+") as f:
        for line in data:
            f.write(line.strip()+"\n")

def main(
    model_dir,
    batch_size,
    tokenizer,
    dataset,
    name,
    data_path,
    data_split,
    lm,
    device,
    log_probs=True):

    if (not os.path.exists(model_dir+"/text.txt")) or (not os.path.exists(model_dir+"/trans.txt")):
        logger.info("################### LOAD PROCESSOR ##################")
        processor = Wav2Vec2Processor.from_pretrained(model_dir, tokenizer_type=tokenizer)

        logger.info("################### LOAD DATASETS ##################")
        eval_dataset = get_data(processor, dataset, name, data_split, data_path, batch_size=batch_size)

        logger.info("################### LOAD MODEL ##################")
        model = Wav2Vec2ForCTC.from_pretrained(model_dir)
        model = model.to(device)

        logger.info("################### DECODE SPEECH DATASETS ##################")
        trans = []
        trans_lm = []
        text = []
        
        if lm['do']:
            decoder = KenLMDecoder(
                processor,
                lm['lm_path'],
                lm['lex_path'])
        for data in tqdm(eval_dataset):
            data = data.to(device)
            with torch.no_grad():
                    logits = model(data.input_values, attention_mask=data.attention_mask).logits
                    if log_probs:
                        logits = F.log_softmax(logits, dim=-1)
                    else:
                        logits = F.softmax(logits, dim=-1)
            
            if lm['do']:
                # KenLM search
                lm_tokens, lm_scores = decoder.decode(logits.cpu().detach())
                trans_batch = processor.batch_decode(lm_tokens[0][:])
                trans.append(trans_batch)
            else:
                # Greedy search
                predicted_ids = torch.argmax(logits, dim=-1)
                trans_batch = processor.batch_decode(predicted_ids)
                trans.append(trans_batch)
            
            # Groundtruth
            data.labels[data.labels == -100] = processor.tokenizer.pad_token_id
            text_batch = processor.batch_decode(data.labels, group_tokens=False)
            text.append(text_batch)
            
            write_result(text_batch, model_dir+"/text.txt")
            write_result(trans_batch, model_dir+"/trans.txt")

        trans = [item for sublist in trans for item in sublist]
        text = [item for sublist in text for item in sublist]
    else:
        logger.info('Decode is already performed!')
        with open(model_dir+"/trans.txt") as f:
            trans = f.readlines()
        with open(model_dir+"/text.txt") as f:
            text = f.readlines()

    logger.info('################### COMPUTE METRICS ###################')
    wer = wer_metric.compute(predictions=trans, references=text, chunk_size=1000)
    cer = cer_metric.compute(predictions=trans, references=text, chunk_size=1000)
    
    if lm['do']:
        write_result(["WER_LM="+str(wer*100), "CER_LM="+str(cer*100)], model_dir+"/results.txt")
    else:
        write_result(["WER="+str(wer*100), "CER="+str(cer*100)], model_dir+"/results.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        help='wav2vec2 model path',
        type=str,
        required=True)
    parser.add_argument(
        '--tokenizer',
        help='tokenizer type used for training the model',
        type=str,
        required=True)
    parser.add_argument(
        '--dataset',
        help='dataset (commonvoice, voxforge, etc.)',
        type=str,
        required=True)
    parser.add_argument(
        '--name',
        help='dataset name',
        type=str,
        default=None)
    parser.add_argument(
        '--data',
        type=str,
        help='evaluation corpus name',
        default='test')
    parser.add_argument(
        '--data_path',
        type=str,
        help='evaluation corpus path',
        default="/workspace/output_models/data")
    parser.add_argument(
        '--batch',
        type=int,
        help='batch size',
        default=32)
    parser.add_argument(
        '--device',
        help='set the device to be used for computation',
        type=str,
        default='cpu')
    parser.add_argument(
        '--lm',
        help='Do language model rescoring',
        type=dict,
        default={'do':False, 'lm_path':None, 'lex_path': None})
    eval_args = parser.parse_args()

    assert eval_args.tokenizer in ["sp", "char"], "tokenizer type must be either 'sp' or 'char'."
    assert eval_args.device in ["cpu", "cuda"], "device must be either 'cuda' or 'cpu'."

    logger.info("Evaluation parameters %s", eval_args)
    
    main(
        model_dir=eval_args.model,
        batch_size=eval_args.batch,
        tokenizer=eval_args.tokenizer,
        dataset=eval_args.dataset,
        name=eval_args.name,
        data_path=eval_args.data_path,
        data_split=eval_args.data,
        lm=eval_args.lm,
        device=eval_args.device,
    )
