#!/usr/bin/env python3
import sys
import json
import os

import module.args
from module.args import set_args, set_checkpoint, set_loggers, set_seeds
from module.data_prep import data_prep
from module.trainer import DataCollatorCTCWithPadding, CTCTrainer

from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)
import datasets
import numpy as np




def set_vocab(path):
    print("################### Prepare VOCAB ##################")
    # Prepare Vocab
    if not os.path.exists(path):
        vocab_list = ['a','e','i','o','u','y','b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','z','à','â','ç','è','é','ê','î','ô','ù','û','|','\'','-']
        vocab_dict = {v: k for k, v in enumerate(vocab_list)}
        vocab_dict["<unk>"] = len(vocab_dict)
        vocab_dict["<pad>"] = len(vocab_dict)

        with open(path, "w") as vocab_file:
            json.dump(vocab_dict, vocab_file)
    
    

def main():
    sys.argv = [
        'run.py',
        '--model_name_or_path=facebook/wav2vec2-large-xlsr-53',
        '--dataset_config_name=fr', 
        '--output_dir=/workspace/output_models/fr/wav2vec2-large-xlsr-53', 
        '--cache_dir=/workspace/data/fr', 
        '--num_train_epochs=25', 
        '--per_device_train_batch_size=32', 
        '--per_device_eval_batch_size=32', 
        '--evaluation_strategy=steps', 
        '--learning_rate=3e-4', 
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
        '--gradient_checkpointing', 
        '--do_train', 
        '--do_eval']
    
    model_args, data_args, training_args = set_args()
    set_loggers(training_args)
    last_checkpoint = set_checkpoint(training_args)
    set_seeds(training_args)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    set_vocab(training_args.output_dir+"/vocab.json")
    tokenizer = Wav2Vec2CTCTokenizer(
        training_args.output_dir+"/vocab.json",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
    )
    
    
    # Prepare feature_extractor & processor
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16_000, padding_value=0.0, do_normalize=True, return_attention_mask=True
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    
    
    print("################### DATA PREPARATION ##################")
    train_dataset = data_prep(
        processor,
        'train+validation',
        training_args.per_device_train_batch_size,
        max_samples=30000,
        max_length=16000*15,
        num_workers=13,
        path_dir="/workspace/output_models/data/fr"
    )
    
    eval_dataset = data_prep(
        processor,
        'test',
        training_args.per_device_eval_batch_size,
        max_samples=100,
        num_workers=1
    )
    
    

    print("################### MODEL LOAD ##################")
    # Model load
    model = Wav2Vec2ForCTC.from_pretrained(
        'facebook/wav2vec2-large-xlsr-53',
        cache_dir="/workspace/output_models/facebook-wav2vec2-large-xlsr-53",
        activation_dropout=0.055,
        attention_dropout=0.094,
        hidden_dropout=0.047,
        feat_proj_dropout=0.04,
        apply_spec_augment=True,
        mask_time_prob=0.4,
        layerdrop=0.041,
        gradient_checkpointing=True,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
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
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    print("################### TRAINER LOAD ##################")
    # Initialize our Trainer
    training_args.report_to=[]
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
