# Automatic Speech Recognition Project

This project aims to create a system that can directly transcribe spoken audio via end-to-end Automatic Speech Recognition (ASR) pipeline. 

## Model 
Unofficial implementation of a paper [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/abs/1512.02595). 

The authors present the end-to-end speech recognition system within a deep learning framework which outperforms traditional approaches. They introduce an architecture which includes CNN and RNN processing steps over Mel Spectrograms. The resulting model is trained on a CTC loss to avoid manual alignment of predicted sequence and time dimension. To make the model streamable instead of a bidirectional RNN we use a unidirectional RNN with a Lookahead convolutions. The future context highly benefits the perfomance and is suitable for online setting. On inference stage the predicted sequence can be constructed using either a standard argmax CTC text deocder or a CTC beam search algorithm. The proposed approach is applicable in noisy environments and can be easily scaled to multiple languages. 


## Requirements
All necessary libraries can be downloaded using the command:
```shell
pip install -r requirements.txt
```

## Train
This project utilizes [Hydra](https://hydra.cc) framework, making it easy to customize training parameters and experiment with different settings. \
To start the learning process you should run the command:
```shell
python3 train.py -cn CONFIG_NAME
```
For example, to train using the `pretrain.yaml` configuration:
```shell
python3 train.py -cn pretrain
```
In folder `src/configs` you can find config files. 
There are two train configs for your convenience: `pretrain.yaml` and `finetune.yaml` for pretraining and fine-tuning stages respectively.

## Inference
To start inference (evaluate the model or save predictions) run the command:
```bash
python3 inference.py -cn CONFIG_NAME
```
There is one inference config `inference.yaml`, where you should specify `save_path` for model predictions. 
Metrics are evaluated and printed into stdout afterwards. 


## Dataset
By default inference dataset is set to Librispeech, but you can transcribe arbitrary audio files or evaluate transcribed audio using `CustomDirAudioDataset`. \
Your file tree should look like this:
```bash
YourDirectoryWithUtterances
├── audio
│   ├── UtteranceID1.wav # may be .flac or .mp3
│   ├── UtteranceID2.wav
│   ...
│
└── transcriptions # ground truth, may not exist
    ├── UtteranceID1.txt
    ├── UtteranceID2.txt
    ...
```

Specify path to audio in `src/configs/datasets/custom_dir.yaml` and run the command:
```bash
python3 inference.py -cn transcribe
```
Transcripted audio you can find in `data/saved`.
If you want to evaluate transcripted audio then you should also add path to transcriptions, change datasets to `custom_dir` in `inference.yaml` and run the same command with `inference.yaml` config. 

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
