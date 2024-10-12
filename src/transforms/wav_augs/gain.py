import torch_audiomentations
from torch import Tensor, nn


class Gain(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Args:
            sample_rate (int): sample rate of input waveform.
            p (float): probability of applying transform.
        """
        super().__init__()
        self._aug = torch_audiomentations.Gain(*args, **kwargs)

    def __call__(self, data: Tensor):
        """
        Args:
            data: waveform of size (batch_size, num_channels, num_samples).
        """
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
