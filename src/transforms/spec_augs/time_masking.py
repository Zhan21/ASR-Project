import torchaudio
from torch import Tensor, nn


class TimeMasking(nn.Module):
    def __init__(self, time_mask_param, p):
        super().__init__()
        self._time_augmentation = torchaudio.transforms.TimeMasking(time_mask_param, p=p)

    def forward(self, spectrogram: Tensor):
        return self._time_augmentation(spectrogram)


class NTimeMasking(nn.Module):
    def __init__(self, time_mask_param, p, n):
        """
        Args:
            time_mask_param (int): maximum range of masked timestamps. Indices uniformly sampled from [0, time_mask_param).
            p (float): maximum proportion of time steps that can be masked. Must be within range [0.0, 1.0].
            n (int): number of masks applied.
        """
        super().__init__()
        self._time_augmentations = nn.Sequential(*[TimeMasking(time_mask_param, p=p) for _ in range(n)])

    def forward(self, spectrogram: Tensor):
        """
        Args:
            spectrogram (Tensor): spectrogram batch of size (B X F X T).
        Returns:
            spectrogram (Tensor): masked spectrograms.
        """
        return self._time_augmentations(spectrogram)
