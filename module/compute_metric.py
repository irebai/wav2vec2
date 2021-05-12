import datasets
import re

model_dir="/workspace/output_models/wav2vec2-large-xlsr-53"

wer_metric = datasets.load_metric("wer")
cer_metric = datasets.load_metric("cer")

# Write output
print('read transcription')
with open(model_dir+"/trans.txt") as f:
    trans = f.readlines()

# Write output
print('read text')
with open(model_dir+"/text.txt") as f:
    text = f.readlines()

#Normalize predictions
trans = [re.sub('\.+', '.', t) for t in trans]
trans = [re.sub('\?+', '?', t) for t in trans]
trans = [re.sub('!+', '!', t) for t in trans]
trans = [re.sub(',+', ',', t) for t in trans]

#Remove index
text = [re.sub('^[^ ]* ', '', t) for t in text]
trans = [re.sub('^[^ ]* ', '', t) for t in trans]


print('computer WER')
wer = wer_metric.compute(predictions=trans, references=text, chunk_size=1000)
cer = cer_metric.compute(predictions=trans, references=text, chunk_size=1000)
print("WER=", wer)
print("CER=", cer)


punctuation='[\,\?\.\!]'
text = [re.sub(punctuation, ' ', t) for t in text]
trans = [re.sub(punctuation, ' ', t) for t in trans]

print('computer WER')
wer = wer_metric.compute(predictions=trans, references=text, chunk_size=1000)
cer = cer_metric.compute(predictions=trans, references=text, chunk_size=1000)
print("WER=", wer)
print("CER=", cer)
