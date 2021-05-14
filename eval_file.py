#!/usr/bin/env python3
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)
import torch
import torchaudio


model_dir="/workspace/huggingface_models/wav2vec2-large-xlsr-53-french"
file_path="/workspace/wavs/Abdel_validation_lvcsr.wav"
file_path="/workspace/wavs/hublot_2019_gafa.wav"
#file_path="/workspace/wavs/1.wav"
#file_path="/workspace/wavs/5.wav"

processor = Wav2Vec2Processor.from_pretrained(model_dir)
model = Wav2Vec2ForCTC.from_pretrained(model_dir)

speech_array, sampling_rate = torchaudio.load(file_path)
if sampling_rate != 16000:
    resampler = torchaudio.transforms.Resample(sampling_rate, 16_000)
    speech_array = resampler(speech_array)

speech_array = speech_array.squeeze().numpy()
inputs = torch.tensor([speech_array])

with torch.no_grad():
    logits = model(inputs).logits

predicted_ids = torch.argmax(logits, dim=-1)
predicted_sentence = processor.batch_decode(predicted_ids)

print(predicted_sentence)