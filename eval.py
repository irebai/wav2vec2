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

model_dir="/workspace/output_models/fr/wav2vec2-large-xlsr-53/checkpoint-14700"
batch_size=32

processor = Wav2Vec2Processor.from_pretrained(model_dir)

eval_dataset = data_prep(
    processor,
    'test',
    batch_size,
    max_samples=100,
    num_workers=1
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

print(trans)
print(text)