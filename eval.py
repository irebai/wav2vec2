#!/usr/bin/env python3
from module.data_prep import data_prep
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)
from module.processor import Wav2Vec2Processor_SP
from module.args import set_loggers

import torch
from module.trainer import DataCollatorCTCWithPadding, BatchRandomSampler
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
    split,
    processor,
    batch_size=1,
    data_path="/workspace/output_models/data"
):
    dataset = data_prep(
        processor,
        split,
        batch_size,
        num_workers=1,
        path_dir=data_path
    )
    data_sampler = BatchRandomSampler(dataset, batch_size)
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=data_sampler,
        collate_fn=data_collator
    )

def write_result(
    data,
    file_path
):
    with open(file_path, "w") as f:
        for line in data:
            f.write(line.strip()+"\n")

def main(
    model_dir,
    batch_size,
    tokenizer,
    data_path,
    data_split
):

    if (not os.path.exists(model_dir+"/text.txt")) or (not os.path.exists(model_dir+"/trans.txt")):
        logger.info("################### LOAD PROCESSOR ##################")
        if tokenizer == 'char':
            processor = Wav2Vec2Processor.from_pretrained(model_dir)
        elif tokenizer == 'sp':
            processor = Wav2Vec2Processor_SP.from_pretrained(model_dir, model_file=model_dir+'/1000_unigram.model')
        else:
            raise ValueError("tokenizer type must be either 'char' or 'sp'")

        eval_dataset = get_data(data_split, processor, batch_size, data_path="/workspace/output_models/data")

        logger.info("################### LOAD MODEL ##################")
        model = Wav2Vec2ForCTC.from_pretrained(model_dir)
        model = model.to('cuda')

        logger.info("################### DECODE SPEECH DATASETS ##################")
        trans = []
        text = []
        for data in tqdm(eval_dataset):
            data = data.to('cuda')
            with torch.no_grad():
                    logits = model(data.input_values, attention_mask=data.attention_mask).logits
            predicted_ids = torch.argmax(logits, dim=-1)

            trans.append(processor.batch_decode(predicted_ids))
            data.labels[data.labels == -100] = processor.tokenizer.pad_token_id
            text.append(processor.batch_decode(data.labels, group_tokens=False))

        trans = [item for sublist in trans for item in sublist]
        text = [item for sublist in text for item in sublist]
        write_result(text, model_dir+"/text.txt")
        write_result(text, model_dir+"/trans.txt")
    else:
        logger.info('Decode is already performed!')
        with open(model_dir+"/trans.txt") as f:
            trans = f.readlines()
        with open(model_dir+"/text.txt") as f:
            text = f.readlines()

    logger.info('################### COMPUTE METRICS ###################')
    wer = wer_metric.compute(predictions=trans, references=text, chunk_size=1000)
    cer = cer_metric.compute(predictions=trans, references=text, chunk_size=1000)
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
    eval_args = parser.parse_args()

    logger.info("Evaluation parameters %s", eval_args)

    main(
        model_dir=eval_args.model,
        batch_size=eval_args.batch,
        tokenizer=eval_args.tokenizer,
        data_path=eval_args.data_path,
        data_split=eval_args.data
    )
