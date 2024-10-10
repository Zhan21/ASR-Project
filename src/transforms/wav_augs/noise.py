import torch
from torch import Tensor, nn

from src.transforms.wav_augs.random_apply import RandomApply


class GaussianNoise(nn.Module):
    def __init__(self, max_scale=0.05):
        super().__init__()
        self.max_scale = max_scale

    def __call__(self, data: Tensor):
        max_amplitude = data.abs().max()
        noise = torch.randn(data.size()) * max_amplitude
        scale = torch.rand(1) * self.max_scale
        return data + noise * scale


class RandomGaussianNoise(nn.Module):
    def __init__(self, max_scale=0.05, p=0.2):
        super().__init__()
        self._random_noiser = RandomApply(augmentation=GaussianNoise(max_scale=max_scale), p=p)

    def __call__(self, data: Tensor):
        return self._random_noiser(data)
