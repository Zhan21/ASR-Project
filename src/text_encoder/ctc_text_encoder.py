import multiprocessing
import re
from collections import defaultdict
from string import ascii_lowercase

import torch
from pyctcdecode import build_ctcdecoder
from torchaudio.models.decoder import download_pretrained_files
from tqdm import tqdm

from src.utils.io_utils import ROOT_PATH
import json

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = "^"

    def __init__(
        self,
        alphabet=None,
        use_lm=True,
        fusion_rate=0.5,
        use_bpe=False,
        train_bpe_on=["train-clean-100", "train-clean-360"],
        vocab_size=100,
        **kwargs,
    ):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be set to ascii
            use_lm (bool): if True then LM shallow fusion used.
            fusion_rate (float): weight of lm during shallow fusion.
            use_bpe (bool): if True then BPE vocabulary trained and used.
            train_bpe_on (list[str]): on which parts train BPE tokenizer.
            vocab_size (int): size of BPE vocabulary.
        """
        if use_bpe:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"

            texts = []
            for part in train_bpe_on:
                index_path = data_dir / f"{part}_index.json"
                assert index_path.exists(), "You should download dataset part {part} to train BPE on it"

                with open(index_path) as f:
                    index = json.load(f)
                texts += [item["text"] for item in index]

            print(f"BPE training on {len(texts)} texts samples")

            tokenizer = Tokenizer(model=BPE())
            tokenizer.pre_tokenizer = Whitespace()

            trainer = BpeTrainer(special_tokens=[self.EMPTY_TOK, " "], vocab_size=vocab_size)

            tokenizer.train_from_iterator(texts, trainer)

            self.char2ind = tokenizer.get_vocab()
            self.ind2char = {idx: ch for ch, idx in self.char2ind.items()}

            self.vocab = [ch for idx, ch in sorted(list(self.ind2char.items()), key=lambda x: x[0])]

        else:
            if alphabet is None:
                alphabet = list(ascii_lowercase + " ")

            self.vocab = [self.EMPTY_TOK] + list(alphabet)

            self.ind2char = dict(enumerate(self.vocab))
            self.char2ind = {ch: idx for idx, ch in self.ind2char.items()}

        lm_path = None

        if use_lm:
            lm_path = download_pretrained_files("librispeech-3-gram").lm

            # lm_download_path = ROOT_PATH / "data" / "lm"

            # if not lm_download_path.is_dir():
            #     lm_download_path.mkdir(exist_ok=True, parents=True)

            # lm_path = lm_download_path / lm_path

            # if not lm_path.is_file():
            #     wget.download("https://openslr.elda.org/resources/11/3-gram.arpa.gz", str(lm_download_path))
            # if not vocab_path.is_file():
            #     wget.download("https://openslr.elda.org/resources/11/librispeech-vocab.txt", str(lm_download_path))

        self.decoder = build_ctcdecoder(
            labels=["" if t == self.EMPTY_TOK else t for t in self.vocab],
            kenlm_model_path=lm_path,
            alpha=fusion_rate,
            beta=1.0,  # weight for length score adjustment of during scoring
        )

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'")

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        """
        Decoding predicted tokens with CTC.

        Args:
            inds (list): list of tokens.
        Returns:
            text (str): clear text without empty tokens and repetitions.
        """
        decoded = []
        last_char = self.EMPTY_TOK

        for i in inds:
            ch = self.ind2char[i]
            if ch != last_char:
                if ch != self.EMPTY_TOK:
                    decoded.append(ch)
                last_char = ch

        return "".join(decoded)

    def _ctc_beam_search_python(self, log_probs, log_probs_length, beam_width=100):
        """
        My implementation of CTC decoding on python. Much slower than pyctcdecode version.
        Do not recommended to use. For educational purposes only.
        """
        decoded_texts = []

        for log_probs_line, length in tqdm(zip(log_probs, log_probs_length)):
            probs = log_probs_line.exp()[:length]
            decoded_texts.append(self._ctc_beam_search_instance(probs, beam_width))

        return decoded_texts

    def _ctc_beam_search_instance(self, probs, beam_width=100, return_best=True):
        """
        Performs CTC decoding with beam search over probabilities.
        Expands paths, merges them and truncates top `beam_width` texts with highest probability.

        Args:
            probs (Tensor): array of probabilities (T X V).
            beam_width (int): number of beams.
            return_best (bool): if True returns text with highest probability.
        Returns:
            hypotheses (list[tuple(str, float)] | str): list of pairs (text, text_prob) or one best text.
        """
        assert len(probs.size()) == 2
        assert probs.size(1) == len(self.ind2char)

        paths = {("", self.EMPTY_TOK): 1.0}

        for frame in probs:
            expanded_paths = defaultdict(float)

            for i, char_prob in enumerate(frame):
                for (prefix, last_char), prefix_prob in paths.items():
                    # last_char is equal to either EMPTY_TOK or last char of prefix

                    char = self.ind2char[i]
                    if char != last_char:
                        if char != self.EMPTY_TOK:
                            prefix = prefix + char
                        last_char = char

                    # sum probs because p('bca') = p(bcaa) = p(bca) * p(a) + p(bca) * p(^)
                    expanded_paths[(prefix, last_char)] += prefix_prob * char_prob

            expanded_paths = sorted(expanded_paths.items(), key=lambda x: x[1], reverse=True)
            paths = dict(expanded_paths[:beam_width])

        hypotheses = [(text, text_prob) for (text, _), text_prob in paths.items()]

        if return_best:
            return max(hypotheses, key=lambda x: x[1])[0]

        return sorted(hypotheses, key=lambda x: x[1], reverse=True)

    def ctc_beam_search(self, log_probs, log_probs_length, beam_width=100):
        """
        Performs CTC decoding with beam search over probabilities.
        Expands paths, merges them and truncates top `beam_width` texts with highest probability.
        Always returns top1 result from the beam search.

        Args:
            log_probs (Tensor): array of log probabilities in batch (B X T X V).
            log_probs_length (Tensor): array of sequence lengths in batch (B).
            beam_width (int): number of beams.
        Returns:
            texts_decoded (list[str]): list of all decoded texts in batch.
        """

        with multiprocessing.get_context("fork").Pool() as pool:
            texts_decoded = self.decoder.decode_batch(pool, log_probs.numpy(), beam_width)

        return texts_decoded

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
