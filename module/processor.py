from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor
)
from module.tokenizer import Wav2Vec2CTCTokenizer_SP

class Wav2Vec2Processor_SP(Wav2Vec2Processor):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        tokenizer = Wav2Vec2CTCTokenizer_SP.from_pretrained(pretrained_model_name_or_path, **kwargs)

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)
