import datasets

model_dir="/workspace/output_models/fr/wav2vec2-large-xlsr-53"

wer_metric = datasets.load_metric("wer.py")
cer_metric = datasets.load_metric("cer.py")

# Write output
print('read transcription')
with open(model_dir+"/trans.txt") as f:
    trans = f.readlines()

# Write output
print('read text')
with open(model_dir+"/text.txt") as f:
    text = f.readlines()

print('computer WER')
wer = wer_metric.compute(predictions=trans, references=text, chunk_size=1000)
cer = cer_metric.compute(predictions=trans, references=text, chunk_size=1000)
print("WER=", wer)
print("CER=", cer)



