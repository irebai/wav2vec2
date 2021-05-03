import sentencepiece as spm
import logging
import os


logger = logging.getLogger(__name__)


class SPTokenizer(Wav2Vec2CTCTokenizer):
    def __init__(
        self,
        model_file_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        do_lower_case=False,
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            do_lower_case=do_lower_case,
            word_delimiter_token='_',
            **kwargs,
        )

        self._word_delimiter_token = None

        self.do_lower_case = do_lower_case

        if not os.path.isfile(self.prefix_model_file + ".model"):
            raise ValueError("Tokenizer is not found!")

        logger.info("==== Loading Tokenizer ===")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file_path + ".model")
        
        

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}


    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()
    
    def get_vocab(self) -> Dict:
        raise NotImplemented()
    

def train_sentencepiece(
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
):
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
        logger.info("Train tokenizer with type:" + self.model_type)

        vocab_size = str(vocab_size)
        character_coverage = str(character_coverage)
        max_sentencepiece_length = str(max_sentencepiece_length)
        bos_id = str(bos_id)
        eos_id = str(eos_id)
        pad_id = str(pad_id)
        unk_id = str(unk_id)
        split_by_whitespace = split_by_whitespace

        """Train tokenizer with unsupervised techniques (BPE, Unigram) using
        SentencePiece Library. If you use "char" mode, the SentencePiece
        creates a char dict so the vocab_size attribute is not needed.
        """
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
    else:
        logger.info("Tokenizer is already trained.")