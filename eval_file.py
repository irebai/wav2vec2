#!/usr/bin/env python3
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor
)
from module.model import Wav2Vec2ForCTC

import torch
import torchaudio
import os
import argparse

def decode(
    model_dir,
    file_path,
    device
):
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir)
    model = model.to(device)

    speech_array, sampling_rate = torchaudio.load(file_path)
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16_000)
        speech_array = resampler(speech_array)

    speech_array = speech_array.squeeze().numpy()
    inputs = torch.tensor([speech_array])
    inputs = inputs.to(device)

    with torch.no_grad():
        logits = model(inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentence = processor.batch_decode(predicted_ids)

    print(predicted_sentence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        help='wav2vec2 model path',
        type=str,
        required=True)
    parser.add_argument(
        '--file',
        help='tokenizer type used for training the model',
        type=str,
        required=True)
    parser.add_argument(
        '--device',
        help='set the device to be used for computation',
        type=str,
        default='cpu'
    )
    eval_args = parser.parse_args()

    if not os.path.exists(eval_args.model+"/pytorch_model.bin"):
        raise ValueError("The model file 'pytorch_model.bin' doesn't exist")

    if not os.path.exists(eval_args.file):
        raise ValueError("The provided file "+eval_args.file+" doesn't exist")

    assert eval_args.device in ['cpu', 'cuda'], "The device option must be either 'cpu' or 'cuda'"

    decode(
        model_dir=eval_args.model,
        file_path=eval_args.file,
        device=eval_args.device
    )
