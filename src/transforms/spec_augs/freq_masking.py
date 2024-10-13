import torchaudio
from torch import Tensor, nn


class FrequencyMasking(nn.Module):
    def __init__(self, freq_mask_param):
        super().__init__()
        self._freq_augmentation = torchaudio.transforms.FrequencyMasking(freq_mask_param)

    def forward(self, spectrogram: Tensor):
        return self._freq_augmentation(spectrogram)


class NFrequencyMasking(nn.Module):
    def __init__(self, freq_mask_param, n):
        """
        Args:
            freq_mask_param (int): maximum range of masked frequencies. Indices uniformly sampled from [0, freq_mask_param).
            n (int): number of masks applied.
        """
        super().__init__()
        self._freq_augmentations = nn.Sequential(*[FrequencyMasking(freq_mask_param) for _ in range(n)])

    def forward(self, spectrogram: Tensor):
        """
        Args:
            spectrogram (Tensor): spectrogram batch of size (B X F X T).
        Returns:
            spectrogram (Tensor): masked spectrograms.
        """
        return self._freq_augmentations(spectrogram)
