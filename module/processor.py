from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor
)
from module.tokenizer import Wav2Vec2CTCTokenizer_SP

class Wav2Vec2Processor(Wav2Vec2Processor):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        tokenizer_type = kwargs.pop("tokenizer_type", 'char')
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if tokenizer_type == 'char':
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif tokenizer_type == 'sp':
            tokenizer = Wav2Vec2CTCTokenizer_SP.from_pretrained(pretrained_model_name_or_path, model_file=pretrained_model_name_or_path+'/tokenizer.model', **kwargs)
        else:
            raise ValueError("tokenizer type must be either 'char' or 'sp'")

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)
