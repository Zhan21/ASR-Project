import torch
import torch.nn as nn


class MaskedCNN(nn.Module):
    def __init__(self, sequential):
        super().__init__()
        self.sequential = sequential

    def forward(self, inputs, input_lengths):
        """
        Fills each spectrogram with 0 along sequence dim
        because after nn.Conv2d/nn.MaxPool2d sequence length becomes shorter
        Inputs:
            - inputs [B, 1, F, T] ~ in terms of torch notation [B, C, H, W]
            - input_lengths [B]
        """
        outputs = None

        for module in self.sequential:
            outputs = module(inputs)
            mask = torch.BoolTensor(outputs.size()).fill_(0)
            if outputs.is_cuda:
                mask = mask.cuda()

            output_lengths = self.transform_input_lengths(module, input_lengths)
            for i, new_length in enumerate(output_lengths):
                length = mask[i].shape[-1]
                new_length = new_length.item()

                if length - new_length > 0:
                    mask[i].narrow(dim=2, start=new_length, length=length - new_length).fill_(1)

            outputs = outputs.masked_fill(mask, 0)
            inputs = outputs
            input_lengths = output_lengths

        return outputs, output_lengths

    def transform_input_lengths(self, module, input_lengths):
        if isinstance(module, nn.Conv2d):
            numerator = input_lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
            input_lengths = numerator.float() / float(module.stride[1])
            input_lengths = input_lengths.int() + 1

        elif isinstance(module, nn.MaxPool2d):
            input_lengths = input_lengths / 2

        return input_lengths.int()


class BatchNormRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bidirectional: bool,
        rnn_type: str,
    ):
        super().__init__()
        get_module = {
            "lstm": nn.LSTM,
            "gru": nn.GRU,
            "rnn": nn.RNN,
        }

        self.norm = nn.BatchNorm1d(input_size)
        self.act = nn.ReLU()

        rnn_module = get_module[rnn_type]
        self.rnn_block = rnn_module(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.1,
            bidirectional=bidirectional,
        )

    def forward(self, inputs, input_lengths):  # [B, T, H], [B]
        inputs = self.act(self.norm(inputs.transpose(1, 2)))  # [B, H, T]

        inputs = inputs.transpose(1, 2)  # [B, T, H]
        max_length = inputs.size(1)

        inputs = nn.utils.rnn.pack_padded_sequence(
            inputs, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )  # PackedSequence
        outputs, _ = self.rnn_block(inputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, total_length=max_length, batch_first=True)  # [B, T, H]

        return outputs  # [B, T, H]


class DeepSpeech2(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        spec_dim: int = 128,
        num_rnn_layers: int = 5,
        hidden_size_rnn: int = 512,
        bidirectional: bool = True,
        rnn_type: str = "gru",
    ):
        super().__init__()
        in_channels = 1
        out_channels = 32  # num channels per batch, per time, per frequency

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardtanh(0, 20, inplace=True),
        )  # [B, 1, F, T] -> [B, 32, F/2, T/2] -> [B, 32, F/4, T/2]

        self.conv = MaskedCNN(self.conv)
        output_size_conv = out_channels * (spec_dim // 4)

        self.rnn_layers = nn.ModuleList()
        output_size_rnn = hidden_size_rnn * 2 if bidirectional else hidden_size_rnn

        for ind in range(num_rnn_layers):
            self.rnn_layers.append(
                BatchNormRNN(
                    input_size=output_size_conv if ind == 0 else output_size_rnn,
                    hidden_size=hidden_size_rnn,
                    bidirectional=bidirectional,
                    rnn_type=rnn_type,
                )
            )

        self.fc = nn.Sequential(
            nn.LayerNorm(output_size_rnn),
            nn.Linear(output_size_rnn, n_tokens, bias=False),
        )

    def forward(self, spectrogram, spectrogram_length, **batch):  # [B, F, T], [B]
        spectrogram = spectrogram.unsqueeze(1)  # [B, C=1, F, T]
        outputs, output_lengths = self.conv(spectrogram, spectrogram_length)  # [B, C, F/4, T/2], [B]

        B, C, F, T = outputs.size()
        outputs = outputs.view(B, C * F, T)  # [B, H=C*F/4, T/2]
        outputs = outputs.transpose(1, 2)  # [B, T/2, H]

        for rnn_layer in self.rnn_layers:
            outputs = rnn_layer(outputs, output_lengths)

        outputs = self.fc(outputs)  # [B, T/2, n_tokens]
        outputs = outputs.log_softmax(dim=-1)

        return {"log_probs": outputs, "log_probs_length": output_lengths}
