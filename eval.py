#!/usr/bin/env python3
from module.data_prep import data_prep
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)
import torch
from module.trainer import DataCollatorCTCWithPadding, BatchRandomSampler
from torch.utils.data import DataLoader
from tqdm import tqdm

model_dir="/workspace/output_models/wav2vec2-large-xlsr-53"
batch_size=32

wer_metric = datasets.load_metric("wer")
cer_metric = datasets.load_metric("cer")

processor = Wav2Vec2Processor.from_pretrained(model_dir)

eval_dataset = data_prep(
    processor,
    'test',
    batch_size,
    num_workers=1,
    path_dir="/workspace/output_models/data"
)

model = Wav2Vec2ForCTC.from_pretrained(model_dir)
model = model.to('cuda')

    
data_sampler = BatchRandomSampler(eval_dataset, batch_size)
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

data = DataLoader(
    eval_dataset,
    batch_size=batch_size,
    sampler=data_sampler,
    collate_fn=data_collator
)


print("################### DECODE SPEECH DATASETS ##################")
trans = []
text = []
for _data in tqdm(data):
    _data = _data.to('cuda')
    with torch.no_grad():
            logits = model(_data.input_values, attention_mask=_data.attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)

    trans.append(processor.batch_decode(predicted_ids))
    _data.labels[ _data.labels == -100] = processor.tokenizer.pad_token_id
    text.append(processor.batch_decode(_data.labels, group_tokens=False))

trans = [item for sublist in trans for item in sublist]
text = [item for sublist in text for item in sublist]


# Write output
with open(model_dir+"/trans.txt", "w") as f:
    for i, t in enumerate(trans):
            f.write(str(i)+" "+t.strip()+"\n")

# Write output
with open(model_dir+"/text.txt", "w") as f:
    for i, t in enumerate(text):
            f.write(str(i)+" "+t.strip()+"\n")


print('computer metrics')
wer = wer_metric.compute(predictions=trans, references=text, chunk_size=1000)
cer = cer_metric.compute(predictions=trans, references=text, chunk_size=1000)
print("WER=", wer)
print("CER=", cer)