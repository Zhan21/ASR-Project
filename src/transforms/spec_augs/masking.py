import torch
import torchaudio
from torch import nn


class TimeMasking(nn.Module):
    """
    Apply masking to a spectrogram in the time domain.
    """

    def __init__(self, time_mask_param, p):
        """
        Args:
            time_mask_param (int): maximum range of masked timestamps. Indices uniformly sampled from [0, time_mask_param).
            p (float): maximum proportion of time steps that can be masked. Must be within range [0.0, 1.0].
        """
        super().__init__()
        self._time_augmentation = torchaudio.transforms.TimeMasking(time_mask_param, p=p)

    def forward(self, spectrogram):
        """
        Args:
            spectrogram (Tensor): spectrogram batch of size (B X F X T).
        Returns:
            spectrogram (Tensor): masked spectrograms.
        """
        return self._time_augmentation(spectrogram)


class FrequencyMasking(nn.Module):
    """
    Apply masking to a spectrogram in the frequency domain.
    """

    def __init__(self, freq_mask_param):
        """
        Args:
            freq_mask_param (int): maximum range of masked frequencies. Indices uniformly sampled from [0, freq_mask_param).
        """
        super().__init__()
        self._freq_augmentation = torchaudio.transforms.FrequencyMasking(freq_mask_param)

    def forward(self, spectrogram):
        """
        Args:
            spectrogram (Tensor): spectrogram batch of size (B X F X T).
        Returns:
            spectrogram (Tensor): masked spectrograms.
        """
        return self._freq_augmentation(spectrogram)
