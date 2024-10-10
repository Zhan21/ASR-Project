import random

import torchaudio
from torch import Tensor, nn

from src.transforms.wav_augs.random_apply import RandomApply


class Fade(nn.Module):
    def __init__(self, fade_shape="linear"):
        super().__init__()
        self._fader = torchaudio.transforms.Fade(fade_shape=fade_shape)

    def __call__(self, data: Tensor):
        self._fader.fade_in_len = random.randint(0, data.size(-1) // 2)
        self._fader.fade_out_len = random.randint(0, data.size(-1) // 2)
        return self._fader(data)


class RandomFade(nn.Module):
    def __init__(self, fade_shape="linear", p=0.2):
        super().__init__()
        self._random_fader = RandomApply(augmentation=Fade(fade_shape=fade_shape), p=p)

    def __call__(self, data: Tensor):
        return self._random_fader(data)
