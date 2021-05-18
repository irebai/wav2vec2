import sentencepiece as spm
import logging
import json
import sys
import os
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2CTCTokenizer
from itertools import groupby

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)

class Wav2Vec2CTCTokenizer_SP(Wav2Vec2CTCTokenizer):
    def __init__(
        self,
        vocab_file,
        model_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        do_lower_case=False,
        do_normalize=False,
        word_delimiter_token='▁',
        return_attention_mask=False,
        **kwargs
    ):
        super().__init__(
            vocab_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            word_delimiter_token=word_delimiter_token,
            do_lower_case=do_lower_case,
            do_normalize=do_normalize,
            return_attention_mask=return_attention_mask,
            **kwargs
        )

        if not os.path.isfile(model_file):
            raise ValueError("Tokenizer is not found!")

        logger.info("Loading Tokenizer")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()
    
    def get_vocab(self) -> Dict:
        raise None

    def _tokenize(self, text, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer.
        """
        if self.do_lower_case:
            text = text.upper()
        tokens_list = self.sp.encode_as_pieces(text)
        return tokens_list

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an index (integer) using the vocab."""
        return self.sp.piece_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp.id_to_piece(index)

    def convert_tokens_to_string(
        self, tokens: List[str], group_tokens: bool = True, spaces_between_special_tokens: bool = False
    ) -> str:
        """
        Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
        """
        # group same tokens into non-repeating tokens in CTC style decoding
        if group_tokens:
            tokens = [token_group[0] for token_group in groupby(tokens)]

        # filter self.pad_token which is used as CTC-blank token
        filtered_tokens = list(filter(lambda token: token != self.pad_token, tokens))

        string = self.sp.decode(filtered_tokens).replace(' ⁇ ','<unk>')

        if self.do_lower_case:
            string = string.lower()
        return string
    
    @classmethod
    def train_sentencepiece(
        cls,
        text_file,
        model_dir,
        vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        user_defined_symbols=None,
        max_sentencepiece_length=10,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
        split_by_whitespace=True,
        **kwargs
    ):
        # Special tokens
        bos_token='<s>'
        eos_token='</s>'
        pad_token='<pad>'
        unk_token='<unk>'
        
        
        if model_type not in ["unigram", "bpe", "char"]:
            raise ValueError("model_type must be one of : [unigram, bpe, char]")
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        if not isinstance(vocab_size, int):
            raise ValueError("vocab_size must be integer.")

        prefix_model_file = os.path.join(
            model_dir, str(vocab_size) + "_" + model_type
        )

        if not os.path.isfile(prefix_model_file + ".model"):
            logger.info("Train tokenizer with type:" + model_type)

            vocab_size = str(vocab_size)
            character_coverage = str(character_coverage)
            max_sentencepiece_length = str(max_sentencepiece_length)
            bos_id = str(bos_id)
            eos_id = str(eos_id)
            pad_id = str(pad_id)
            unk_id = str(unk_id)
            split_by_whitespace = split_by_whitespace

            query = (
                "--input="
                + text_file
                + " --model_prefix="
                + prefix_model_file
                + " --model_type="
                + model_type
                + " --bos_id="
                + bos_id
                + " --eos_id="
                + eos_id
                + " --pad_id="
                + pad_id
                + " --unk_id="
                + unk_id
                + " --max_sentencepiece_length="
                + max_sentencepiece_length
                + " --character_coverage="
                + character_coverage
            )
            if model_type not in ["char"]:
                # include vocab_size
                query += " --vocab_size=" + str(vocab_size)
            if user_defined_symbols is not None:
                query += " --user_defined_symbols=" + user_defined_symbols
            if not split_by_whitespace:
                query += " --split_by_whitespace=false"
            
            # Train tokenizer
            spm.SentencePieceTrainer.train(query)
            
            # Save Train query
            with open(prefix_model_file + ".query", "w") as f:
                f.write(query)
                
            # Save special tokens
            vocab_dict = {}
            vocab_dict[bos_token]=bos_id
            vocab_dict[eos_token]=eos_id
            vocab_dict[pad_token]=pad_id
            vocab_dict[unk_token]=unk_id
            vocab_dict = {k: v for k, v in vocab_dict.items() if v != -1}
            with open(prefix_model_file+".tokens.json", "w") as vocab_file:
                json.dump(vocab_dict, vocab_file)

        else:
            logger.info("Tokenizer is already trained.")
        
        return cls(
            prefix_model_file+".tokens.json",
            prefix_model_file + ".model",
            **kwargs
        )



class Wav2Vec2CTCTokenizer_CHAR(Wav2Vec2CTCTokenizer):
    @classmethod
    def set_vocab(
        cls,
        vocab_file,
        bos_token=None,
        eos_token=None,
        do_punctuation=False,
        **kwargs
    ):
        print("################### Prepare VOCAB ##################")
        # Prepare Vocab
        word_delimiter_token = '|'
        vocab_list = ['a','e','i','o','u','y','b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','z']
        vocab_list += ['à','â','æ','ç','è','é','ê','ë','î','ï','ô','œ','ù','û','ü','ÿ']
        vocab_list += ["'",'-']
        vocab_list += [word_delimiter_token]

        if do_punctuation:
            vocab_list += ['.',',','!','?']

        vocab_dict = {v: k for k, v in enumerate(vocab_list)}
        vocab_dict["<unk>"] = len(vocab_dict)
        vocab_dict["<pad>"] = len(vocab_dict)
        if bos_token is not None:
            vocab_dict["<bos>"] = len(vocab_dict)
        if eos_token is not None:
            vocab_dict["<eos>"] = len(vocab_dict)

        with open(vocab_file, "w") as file:
            json.dump(vocab_dict, file)

        return cls(
            vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token="<unk>",
            pad_token="<pad>",
            word_delimiter_token="|",
            do_lower_case=False,
            **kwargs
        )

