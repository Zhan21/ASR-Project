from pathlib import Path

import pandas as pd

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch, examples_to_log=10)
            self.log_predictions(**batch, examples_to_log=10)
            self.log_audio(**batch, examples_to_log=1)
        else:
            self.log_spectrogram(**batch, examples_to_log=10)
            self.log_predictions(**batch, examples_to_log=10)
            self.log_audio(**batch, examples_to_log=1)

    def log_audio(self, audio, audio_path, examples_to_log=1, **batch):
        unique_audio = {}
        for wav, wav_path in zip(audio, audio_path):
            unique_audio[Path(wav_path).name] = wav

        audio_to_log = list(unique_audio.values())[:examples_to_log]
        for i, wav in enumerate(audio_to_log):
            self.writer.add_audio(f"audio/{i+1}/part", wav, sample_rate=16000)

    def log_spectrogram(self, spectrogram, examples_to_log=10, **batch):
        spectrogram = spectrogram.detach().cpu()[:examples_to_log]
        spectrogram_img = plot_spectrogram(spectrogram)
        self.writer.add_image("spectrogram/part", spectrogram_img)

    def log_predictions(self, text, log_probs, log_probs_length, audio_path, examples_to_log=10, **batch):
        # TODO add beam search
        # Note: by improving text encoder and metrics design
        # this logging can also be improved significantly

        log_probs = log_probs[:examples_to_log].detach().cpu()
        log_probs_length = log_probs_length[:examples_to_log].detach().cpu()

        argmax_inds = log_probs.argmax(-1)
        texts_argmax_raw = []
        texts_argmax = []

        for inds, ind_len in zip(argmax_inds, log_probs_length):
            inds = inds[:ind_len].numpy()
            texts_argmax_raw.append(self.text_encoder.decode(inds))
            texts_argmax.append(self.text_encoder.ctc_decode(inds))

        texts_beamsearch = self.text_encoder.ctc_beam_search(log_probs, log_probs_length, beam_width=100)

        tuples = list(zip(text, texts_argmax_raw, texts_argmax, texts_beamsearch, audio_path))

        rows = {}
        for target, pred_raw, pred, pred_beamsearch, audio_path in tuples:
            target = self.text_encoder.normalize_text(target)
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100
            wer_beamsearch = calc_wer(target, pred_beamsearch) * 100
            cer_beamsearch = calc_cer(target, pred_beamsearch) * 100

            rows[Path(audio_path).name] = {
                "target": target,
                "raw argmax": pred_raw,
                "argmax": pred,
                "beamsearch": pred_beamsearch,
                "wer": wer,
                "cer": cer,
                "wer_beamsearch": wer_beamsearch,
                "cer_beamsearch": cer_beamsearch,
                "audio path": audio_path,
            }
        self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))
