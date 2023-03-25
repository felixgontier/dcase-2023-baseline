# Baseline system for DCASE 2023 task 6, subtask A

This repository contains the baseline system for the DCASE 2023 challenge task 6A on audio captioning.

The main model is composed of a convolutional encoder and a transformer decoder, that autoregressively models captions conditionally to log-mel spectrograms. This year, the baseline reuses the audio encoder trained as part of the [task 6B baseline](https://github.com/xieh97/dcase2023-audio-retrieval) on audio retrieval.

For more information, please refer to the corresponding [DCASE subtask page](https://dcase.community/challenge2023/task-automatic-audio-captioning).

----

## Table of contents

 1. [Repository setup](#repository-setup)
 2. [Clotho dataset](#clotho-dataset)
    1. [Obtaining the data from Zenodo](#obtaining-the-data-from-zenodo)
    2. [Data pre-processing](#data-pre---processing)
    3. [Pre-processing parameters](#pre---processing-parameters)
 3. [Running the baseline system](#running-the-baseline-system)
    1. [Running an experiment](#running-an-experiment)
    2. [Evaluation with pre-trained weights](#evaluation-with-pre---trained-weights)
 4. [Details of experiment settings](#details-of-experiment-settings)
    1. [Adaptation settings](#adaptaion-settings)
    2. [Data settings](#data-settings)
    3. [Language model settings](#language-model-settings)
    4. [Training settings](#training-settings)
    5. [Workflow settings](#workflow-settings)

----

## Repository setup

The first step in running the baseline system is to clone this repository on your computer:

````shell script
$ git clone git@github.com:felixgontier/dcase-2023-baseline.git
````

This operation will create a `dcase-2023-baseline` directory at the current location, with the contents of this repository. The `dcase-2023-baseline` will be referred to as the root directory in the rest of this readme.

Next, a recent version of PyTorch is required to run the baseline.

**Note**: The baseline system is developed with Python 3.7, PyTorch 1.7.1 and CUDA 10.1.
Please refer to the [PyTorch setup guide](https://pytorch.org/get-started/locally/) for PyTorch/CUDA compatibility information.

Other required packages can be installed using Pip by running the following command in the root directory:

````shell script
$ python3.7 -m venv env/ # Optionally create a virtual environment
$ pip install -r requirements_pip.txt
````

The audio encoder of the baseline system is initialized with trained weights from the retrieval subtask baseline. The corresponding checkpoint `audio_encoder.pth` is hosted on [Zenodo](https://zenodo.org/record/7752975).

 1. Download `audio_encoder.pth` from the Zenodo repository and place it in the root baseline directory.

Lastly, the [caption-evaluation-tools](https://github.com/audio-captioning/caption-evaluation-tools) is needed for evaluation.

 1. Download and extract the repository in the baseline root directory.
 2. Download the Stanford models by running:

````shell script
$ cd coco_caption
$ ./get_stanford_models.sh
````

**Note** that the caption evaluation tools require that Java is installed and enabled.

----

## Clotho dataset

### Obtaining the data from Zenodo

The Clotho v2.1 dataset can be found on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4783391.svg)](https://doi.org/10.5281/zenodo.4783391)

The test set (without captions) is available separately: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3865658.svg)](https://doi.org/10.5281/zenodo.3865658)

After downloading all `.7z` archives and `.csv` caption files from both repositories, audio files should be extracted in the `data` directory.

Specifically, the directory structure should be as follows from the baseline root directory:

    data/
     | - clotho_v2/
     |   | - development/
     |   |   | - *.wav
     |   | - validation/
     |   |   | - *.wav
     |   | - evaluation/
     |   |   | - *.wav
     |   | - test/
     |   |   | - *.wav
     |   | - clotho_captions_development.csv
     |   | - clotho_captions_validation.csv
     |   | - clotho_captions_evaluation.csv

### Data pre-processing

Pre-processing operations are implemented in `clotho_dataset.py` and `audio_logmels.py`. These are the same as for the [task 6B baseline](https://github.com/xieh97/dcase2023-audio-retrieval), except for the addition of the Clotho-testing subset handling.

Dataset preparation is done by running the following command:

````shell script
$ python clotho_dataset.py
$ python audio_logmels.py
````

The script outputs `<split>_audio_logmels.hdf5` and `<split>_text.csv` files in the `data` subdirectory.


----

## Running the baseline system

### Running an experiment

Experiments settings are defined in a YAML file located in the `exp_settings` directory. The `dcb.yaml` file contains parameters used to produce the reported baseline results.
Specific settings are detailed below.

To run an experiment according to a `<exp_name>.yaml` settings file, use the following command:

````shell script
$ python main.py --exp <exp_name>
````

After training, model weights are saved to a `outputs/<exp_name>_out/` directory.

### Evaluation with pre-trained weights

 1. Download pre-trained weights from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7688773.svg)](https://doi.org/10.5281/zenodo.7688773)
 2. In `exp_settings/dcb.yaml`, change the `lm/eval_model` setting to `/path/to/dcase_baseline_pre_trained.bin`, with the correct path to the downloaded file.
 3. Set the `workflow/train` and `workflow/validate` to `false`, and `workflow/evaluate` and/or `workflow/infer` to `true`.
 4. Run the evaluation and/or inference.

````shell script
$ python main.py --exp dcb
````

----

## Details of experiment settings

Experiment settings described in the `exp_settings/dcb.yaml` file are:

    adapt:
      audio_emb_size: 2048
      nb_layers: 1
    data:
      root_dir: data
      max_audio_len: 2048
      max_caption_tok_len: 64
    lm:
      config: # Model parameters
        activation_dropout: 0.1
        activation_function: 'gelu'
        attention_dropout: 0.1
        classifier_dropout: 0.0
        d_model: 768
        decoder_attention_heads: 12
        decoder_ffn_dim: 3072
        decoder_layers: 6
        dropout: 0.1
        encoder_attention_heads: 12
        encoder_ffn_dim: 3072
        encoder_layers: 0
        vocab_size: 50265
      generation: # Generation parameters
        early_stopping: true
        no_repeat_ngram_size: 3
        num_beams: 4
        min_length: 5
        max_length: 100
        length_penalty: 1.0
        decoding: beam
      eval_model: best
      eval_checkpoint: null
      freeze:
        all: false
        attn: false
        dec: false
        dec_attn: false
        dec_mlp: false
        dec_self_attn: false
        enc: false
        enc_attn: false
        enc_mlp: false
        mlp: false
      tokenizer: facebook/bart-base
      pretrained: null
    training:
      eval_steps: 1000
      force_cpu: false
      batch_size: 4
      gradient_accumulation_steps: 2
      num_workers: 8
      lr: 1.0e-05
      nb_epochs: 20
      save_steps: 1000
      seed: 0
    workflow:
      train: true
      validate: true
      evaluate: true
      infer: false

### Adaptation settings

The `adaptation` block defines a small adaptation network before the transformer encoder. Its aim is to adjust the dimension of audio features to that of the transformer (`lm/config/d_model` setting).

 * `audio_emb_size` (int): Dimension of audio features, i.e. the input dimension of the adaptation network. In the case of VGGish embeddings, this setting is set to 128.
 * `nb_layers` (int): Number of layers of the network. If set to 0, the dimension of audio features must be equal to that of the transformer. If greater than 1, the network will contain `nb_layers` dense layers with output dimension `lm/config/d_model` and ReLU activations. The last layer of the adaptation network has no activation function.

### Data settings

The `data` block contains settings related to the dataset.

 * `root_dir` (str): Path to the data root directory.
 * `max_audio_len` and `max_caption_tok_len` (int): The data loader pads each example audio and tokenized caption to a set duration for batching. Provided values are adapted to the VGGish representation and BART tokenization of the baseline.

### Language model settings

The `lm` block contains settings related to both the encoder and decoder of the main transformer model, which is derived from BART.

The `config` sub-block details the model, as per the [HuggingFace BART configuration](https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/bart#transformers.BartConfig). Provided settings replicate the bart-base model configuration.

**Note**: The `vocab_size` parameter depends on the pre-trained tokenizer defined by `lm/tokenizer`.

The `generation` sub-block provides generation-specific settings (see the [HuggingFace Generation documentation](https://huggingface.co/docs/transformers/master/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin)):

 * `decoding` (str): `beam` or `greedy` decoding are supported.

The `freeze` sub-block enables freezing different components of the transformer (attention, MLP, self-attention or cross-attention).

Other parameters are:

 * `eval_model` (str): Model selection at evaluation/inference. `best` selects the best model according to validation loss at training, `checkpoint` uses a specific checkpoint set by `eval_checkpoint`. This setting can also be set to `/path/to/model.bin` for custom trained model weights, e.g. the provided pre-trained weights.
 * `eval_checkpoint` (int): Model checkpoint to use at evaluation/inference. This is ignored unless `eval_model` is set to `checkpoint`.
 * `tokenizer` (str): Name of the HuggingFace pre-trained tokenizer.
 * `pretrained` (str, null): If not null, name of a HuggingFace pre-trained model (e.g. facebook/bart-base). **Note** that this will bypass all `config` sub-block settings.

### Training settings

The `training` block describes parameters of the training process.

 * `eval_steps` (int): Frequence of model validation, in training steps.
 * `save_steps` (int): Frequence of model weights saving, in training steps. If `lm/eval_model` is set to `best`, this should be a factor of `eval_steps`.
 * `force_cpu` (bool): Force all computations on CPU, even when CUDA is available.
 * `batch_size` (int): Batch size during model training and validation.
 * `gradient_accumulation_steps` (int): Accumulates gradients over several steps, effectively increasing the batch size without additional memory cost. Gradient accumulation is disabled if this is set to 1.
 * `num_workers` (int):  Number of CPU workers for data loading.
 * `lr` (float): Learning rate during training.
 * `nb_epochs` (int): Number of training epochs.
 * `seed` (int, null): Sets a specific torch random seed before experiments. **Note** that this does not ensure reproducibility when training on GPU.

### Workflow settings

The `workflow` block sets operations to be conducted in the experiment.

 * `train` will perform optimization with data in the `</path/to/data>/development` directory, where `</path/to/data>` is the appended `data/root_dir` and `data/features_dir` settings.
 * `validate` must be set to `true` during training if `lm/eval_model` is set to `best`. Validation is done on data in the `</path/to/data>/validation` directory.
 * `evaluate` refers to evaluation with metrics, and outputs `metrics_coco_<decoding_method>.json` and `generated_captions_<decoding_method>.txt` files in the `output/<exp_name>_out` directory, where `<decoding_method>` is the `lm/generation/decoding` setting. Evaluation is done on data in the `</path/to/data>/evaluation` directory.
 * `infer` refers to caption generation without computing metrics. Inference outputs a submission-ready file `test_output_captions_<decoding_method>.csv`. Inference is performed on data in the `</path/to/data>/test` directory.



