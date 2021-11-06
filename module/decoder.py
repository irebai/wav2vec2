from flashlight.lib.sequence.criterion import CpuViterbiPath, get_data_ptr_as_bytes
from flashlight.lib.text.dictionary import create_word_dict, load_words
from flashlight.lib.text.decoder import (
        CriterionType,
        LexiconDecoderOptions,
        KenLM,
        LM,
        LMState,
        SmearingMode,
        Trie,
        LexiconDecoder)
from flashlight.lib.text.decoder import LexiconFreeDecoder, LexiconFreeDecoderOptions
import math
import torch
import numpy as np

class KenLMDecoder(object):
    def __init__(self,
                 processor,
                 kenlm_model_path,
                 lexicon_path = None,
                 kenlm_args = {
                    "beam": 200,
                    "nbest": 1,
                    "beam_threshold": 20,
                    "lm_weight": 1.0,
                    "word_score": 1.0,
                    "sil_weight": 0}):

        vocab_dict = processor.tokenizer.get_vocab()
        
        pad = processor.tokenizer.pad_token
        unk = processor.tokenizer.unk_token
        silence = processor.tokenizer.word_delimiter_token
        

        self.vocab_size = len(vocab_dict)
        self.blank_token = (vocab_dict[pad])
        self.silence_token = vocab_dict[silence]
        self.unk_token = vocab_dict[unk]

        self.nbest = kenlm_args['nbest']

        if lexicon_path:
            vocab_keys = vocab_dict.keys()
            self.lexicon = load_words(lexicon_path)
            self.word_dict = create_word_dict(self.lexicon)
            self.unk_word = self.word_dict.get_index(unk)

            self.lm = KenLM(kenlm_model_path, self.word_dict)
            self.trie = Trie(self.vocab_size, self.silence_token)

            start_state = self.lm.start(False)
            for i, (word, spellings) in enumerate(self.lexicon.items()):
                word_idx = self.word_dict.get_index(word)
                _, score = self.lm.score(start_state, word_idx)

                for spelling in spellings:
                    spelling_idxs = []
                    for token in spelling:
                        if token.upper() in vocab_keys:
                            spelling_idxs.append(vocab_dict[token.upper()])
                        elif token.lower() in vocab_keys:
                            spelling_idxs.append(vocab_dict[token.lower()])
                        else:
                            print("WARNING: The token", token, "not exist in your vocabulary, using <unk> token instead")
                            spelling_idxs.append(self.unk_token)
                    self.trie.insert(spelling_idxs, word_idx, score)
            self.trie.smear(SmearingMode.MAX)

            self.decoder_opts = LexiconDecoderOptions(
                beam_size=kenlm_args['beam'],
                beam_size_token=kenlm_args['beam_size_token'] if "beam_size_token" in kenlm_args else len(vocab_dict),
                beam_threshold=kenlm_args['beam_threshold'],
                lm_weight=kenlm_args['lm_weight'],
                word_score=kenlm_args['word_score'],
                unk_score=-math.inf,
                sil_score=kenlm_args['sil_weight'],
                log_add=False,
                criterion_type=CriterionType.CTC,
            )

            self.decoder = LexiconDecoder(
                self.decoder_opts,
                self.trie,
                self.lm,
                self.silence_token,
                self.blank_token,
                self.unk_word,
                [],
                False,
            )
        else:
            d = {w: [[w]] for w in vocab_dict.keys()}
            self.word_dict = create_word_dict(d)
            self.lm = KenLM(kenlm_model_path, self.word_dict)
            self.decoder_opts = LexiconFreeDecoderOptions(
                beam_size=kenlm_args['beam'],
                beam_size_token=kenlm_args['beam_size_token'] if "beam_size_token" in kenlm_args else len(vocab_dict),
                beam_threshold=kenlm_args['beam_threshold'],
                lm_weight=kenlm_args['lm_weight'],
                sil_score=kenlm_args['sil_weight'],
                log_add=False,
                criterion_type=CriterionType.CTC,
            )
            self.decoder = LexiconFreeDecoder(
                self.decoder_opts, self.lm, self.silence_token, self.blank_token, []
            )

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank"""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank_token, idxs)
        return torch.LongTensor(list(idxs))

    def decode(self, emissions):
        B, T, N = emissions.size()
        # print(emissions.shape)
        tokens = []
        scores = []
        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, T, N)
            nbest_results = results[: self.nbest]
            tokens_nbest = []
            scores_nbest = []
            for result in nbest_results:
                tokens_nbest.append(result.tokens)
                scores_nbest.append(result.score)
            tokens.append(tokens_nbest)
            scores.append(scores_nbest)

        token_array = np.array(tokens, dtype=object).transpose((1, 0, 2))
        scores_arrray = np.array(scores, dtype=object).transpose()
        return token_array, scores_arrray

