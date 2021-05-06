#!/usr/bin/env python3
from module.text_map import normalize_text
import torchaudio
import json
import datasets
import os
import multiprocessing


# Create and save tokenizer
def remove_special_characters(batch):
    batch["target_text"] = normalize_text(batch["sentence"]) + " "
    batch["text_len"] = len(batch["target_text"])
    return batch

# Preprocessing the datasets.
# We need to read the aduio files as arrays and tokenize the targets.
def speech_file_to_array_fn(batch):
    resampler = torchaudio.transforms.Resample(48_000, 16_000)
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    batch["sampling_rate"] = 16_000
    batch["speech_len"] = len(batch["speech"])
    return batch

def load_data(data, save_dir="/workspace/output_models/data/fr"):  
    # Get the datasets:
    print("################### LOAD DATASETS ##################")
    return datasets.load_dataset("common_voice", "fr", split=data, cache_dir=save_dir)

def prepare_text(dataset):
    print("################### PREPARE TEXT DATASETS ##################")
    return dataset.map(remove_special_characters, remove_columns=["sentence"])

def select_subset(dataset, nb_samples):
    print("################### SELECT SUBSETS FROM DATASETS ##################")
    return dataset.select(range(nb_samples))

def prepare_speech(dataset, num_workers=None):
    print("################### GET SPEECH DATASETS ##################")
    return dataset.map(
        speech_file_to_array_fn,
        num_proc=multiprocessing.cpu_count() if num_workers == None else num_workers,
    )

def get_final_data(dataset, batch_size, processor, num_workers=None):
    print("################### GET INPUT & LABEL DATASETS ##################")
    def prepare_dataset(batch):
        # check that all files have the correct sampling rate
        assert (
            len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
        batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
        # Setup the processor for targets
        with processor.as_target_processor():
            batch["labels"] = processor(batch["target_text"]).input_ids
        return batch

    return dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names,
        batch_size=batch_size,
        batched=True,
        num_proc=multiprocessing.cpu_count() if num_workers == None else num_workers,
    )

def data_filter(dataset, param, max_len):
    print("################### FILTER DATASETS ##################")
    fn = lambda data: (data[param] < max_len)
    return dataset.filter(fn)

def data_sort(dataset, param):
    print("################### FILTER DATASETS ##################")
    return dataset.sort(column=param)

def write_text(dataset, param, output="/workspace/output_models/text_asr"):
    print("################### WRITE TEXT ##################")
    text = [data[param] for data in dataset]
    with open(output, "w") as f:
        for i, t in enumerate(text):
             f.write(str(i)+" "+t.strip()+"\n")

def datasets_concat(dataset1, dataset2):
    print("################### CONCAT TRAIN DATASETS ##################")
    return datasets.concatenate_datasets([dataset1, dataset2])


def data_prep(
    processor,
    split,
    batch_size,
    max_samples=None,
    max_length=None,
    filter_and_sort_param='speech_len',
    path_dir="/workspace/output_models/data/fr",
    num_workers=1):

    #load data
    data = load_data(split, save_dir=path_dir)
    
    #select subset
    if max_samples is not None:
        data = select_subset(data, max_samples)
    
    #prepare speech (since it doesn't change)
    data = prepare_speech(data, num_workers=num_workers)
    
    #prepare text
    data = prepare_text(data)
    
    #filter speech
    if max_length is not None:
        data = data_filter(data, filter_and_sort_param, max_length)
    
    #sort speech
    data = data_sort(data, filter_and_sort_param)
    
    #get supervised data
    data = get_final_data(data, batch_size, processor, num_workers=num_workers)
    
    return data