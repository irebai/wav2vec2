#!/usr/bin/env python3
import sys
import json
import os
import logging

import module.args
from module.args import set_args, set_checkpoint
from module.data_prep import data_prep, get_text
from module.trainer import DataCollatorCTCWithPadding, CTCTrainer
from module.tokenizer import (
    Wav2Vec2CTCTokenizer_CHAR,
    Wav2Vec2CTCTokenizer_SP,
)
from module.model import Wav2Vec2ForCTC
from module.augmentation import SpeechAugment
from transformers.trainer_utils import is_main_process
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    set_seed
)
import datasets
import numpy as np

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)

def main():
    # output directory to save models and data
    output_dir="/workspace"
    # Vocab list
    vocab_list = ['a','e','i','o','u','y','b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','z']
    vocab_list += ['à','â','æ','ç','è','é','ê','ë','î','ï','ô','œ','ù','û','ü','ÿ']
    vocab_list += [' ',"'",'-']
    # special tokens
    unk_id=0
    pad_id=1
    # SP train text
    train_text=output_dir + '/train.txt'


    model_args, data_args, training_args = set_args([
        '--model_name_or_path=facebook/wav2vec2-large-xlsr-53',
        '--dataset_config_name=fr', 
        '--output_dir='+output_dir+'/output_models/wav2vec2-large-xlsr-53',
        '--cache_dir='+output_dir+'/output_models/data',
        '--num_train_epochs=25', 
        '--per_device_train_batch_size=32', 
        '--per_device_eval_batch_size=32', 
        '--evaluation_strategy=steps', 
        '--learning_rate=1e-4',
        '--warmup_steps=500', 
        '--fp16', 
        '--overwrite_output_dir',
        '--freeze_feature_extractor', 
        '--save_steps=100', 
        '--eval_steps=100', 
        '--save_total_limit=1', 
        '--logging_steps=100', 
        '--feat_proj_dropout=0.0', 
        '--layerdrop=0.1', 
        '--tokenizer_type=char',
        '--gradient_checkpointing', 
        '--do_train', 
        '--do_eval'])
    last_checkpoint = set_checkpoint(training_args)

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)


    logger.info("################### PROCESSOR PREPARATION ##################")
    # Prepare tokenizer
    assert model_args.tokenizer_type in ["sp", "char"], "tokenizer type must be either 'sp' or 'char'."

    if model_args.tokenizer_type == 'char':
        tokenizer = Wav2Vec2CTCTokenizer_CHAR.set_vocab(
            training_args.output_dir+"/vocab.json",
            vocab_list,
            do_punctuation=False,
            unk_id=unk_id,
            pad_id=pad_id,
        )
    elif model_args.tokenizer_type == 'sp':
        if not os.path.exists(train_text):
            get_text('common_voice', 'fr', 'train', train_text, path_dir=model_args.cache_dir)
        tokenizer = Wav2Vec2CTCTokenizer_SP.train_sentencepiece(
            train_text,
            output_dir + '/tokenizer_new/',
            1000,
            model_type="unigram",
            pad_id=1,
            unk_id=0,
            vocab=vocab_list
        )
    # Prepare feature_extractor
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16_000, padding_value=0.0, do_normalize=True, return_attention_mask=True
    )
    # Prepare processor
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)    

    logger.info("################### DATA PREPARATION ##################")
    train_dataset = data_prep(
        processor,
        'common_voice',
        'fr',
        'train+validation',
        training_args.per_device_train_batch_size,
        path_dir=model_args.cache_dir,
        max_samples=100000,
        max_length=16000*15,
        vocab=vocab_list
    )
    eval_dataset = data_prep(
        processor,
        'common_voice',
        'fr',
        'test',
        training_args.per_device_eval_batch_size,
        path_dir=model_args.cache_dir,
        max_samples=100,
    )
    
    logger.info("################### MODEL LOAD ##################")
    # Model load
    model = Wav2Vec2ForCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=output_dir + "/output_models/facebook-wav2vec2-large-xlsr-53",
        activation_dropout=0.055,
        attention_dropout=0.094,
        hidden_dropout=0.047,
        feat_proj_dropout=0.04,
        apply_spec_augment=True,
        mask_time_prob=0.2,
        layerdrop=0.041,
        gradient_checkpointing=True,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        time_pooling_size=1,
        pooling_type="max",
        normalize_wav2vec=False,
        normalize_type="batch",
        num_ff_layers=0,
        reduce_ff_layer=1,
    )
    
    if model_args.freeze_feature_extractor:
        model.freeze_feature_extractor()


    
    # Metric
    wer_metric = datasets.load_metric("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        
        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    

    # Data collator
    augmenter = SpeechAugment(
        noise_dir=output_dir + '/noise/background_noises',
        rir_dir=output_dir + '/noise/',
        rir_lists=['simulated_rirs_16k/smallroom/rir_list', 'simulated_rirs_16k/mediumroom/rir_list', 'simulated_rirs_16k/largeroom/rir_list'],
        apply_prob=0.2,
        sample_rate=16000,
    )
    data_collator = DataCollatorCTCWithPadding(processor=processor, augmenter=augmenter, padding=True)

    logger.info("################### TRAINER LOAD ##################")
    # Initialize our Trainer
    training_args.report_to=[]
    # Set seed before initializing model.
    set_seed(training_args.seed)
    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=processor.feature_extractor,
    )
  


    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        # save the feature_extractor and the tokenizer
        if is_main_process(training_args.local_rank):
            processor.save_pretrained(training_args.output_dir)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    return results



if __name__ == "__main__":
    main()
