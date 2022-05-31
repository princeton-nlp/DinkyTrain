<p align="center">
  <!--<a href="https://github.com/fairseq/fairseq"><img src="docs/pnlp_logo512.png" width="150"></a>-->
  <a href="https://github.com/princeton-nlp/DinkyTrain"><img src="docs/DinkyTrainLogo.png" width="200"></a>
  <!--<a href="https://princeton-nlp.github.io/"><img src="docs/fairseq_logo.png" width="150"></a>-->
  <br />
</p>

This repository provides a library for efficient training of masked language models (MLM), built with [fairseq](https://github.com/fairseq/fairseq). We fork fairseq to give researchers more flexibility when using our training scripts,
while also making it easier to adapt our code contributions into other projects.

## Why DinkyTrain?
The [Dinky](https://en.wikipedia.org/wiki/Princeton_Branch) runs between Princeton Junction and Princeton and is the shortest scheduled commuter rail line in the United States.
We also aim to make pre-training short and accessible to everyone.

## Our Contributions
* [DeepSpeed transformer kernel](https://www.deepspeed.ai/tutorials/transformer_kernel/) integration
* A training recipe for efficient MLM pre-training
* An easy-to-follow guideline of using fairseq for MLM pre-training.

Other [fairseq features](https://github.com/fairseq/fairseq#features):
* Multi-GPU training on one machine or across multiple machines (data and model parallel)
* [Gradient accumulation](https://fairseq.readthedocs.io/en/latest/getting_started.html#large-mini-batch-training-with-delayed-updates) enables training with large mini-batches even on a single GPU
* [Mixed precision training](https://fairseq.readthedocs.io/en/latest/getting_started.html#training-with-half-precision-floating-point-fp16) (trains faster with less GPU memory on [NVIDIA tensor cores](https://developer.nvidia.com/tensor-cores))
* [Extensible](https://fairseq.readthedocs.io/en/latest/overview.html): easily register new models, criterions, tasks, optimizers and learning rate schedulers
* [Flexible configuration](docs/hydra_integration.md) based on [Hydra](https://github.com/facebookresearch/hydra) allowing a combination of code, command-line and file based configuration
* [Full parameter and optimizer state sharding](examples/fully_sharded_data_parallel/README.md)
* [Offloading parameters to CPU](examples/fully_sharded_data_parallel/README.md)

See the [fairseq repo](https://github.com/fairseq/fairseq) and its [documentation](https://fairseq.readthedocs.io/) for more details on how to use and extend fairseq.

# DinkyTrain for Efficient MLM Pre-training

## Quick Links

  - [Overview](#overview)
  - [Installation](#installation)
  - [Data Pre-processing](#data-pre-processing)
  - [Pre-training](#pre-training)
  - [Fine-tuning on GLUE and SQuAD](#fine-tuning-on-glue-and-squad)
  - [Convert to HuggingFace](#convert-to-huggingface)
  - [Model List](#model-list)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)

## Overview
You can reproduce the pre-training experiments of our recent paper [Should You Mask 15% in Masked Language Modeling?](https://arxiv.org/abs/2202.08005),
where we find that higher masking rates can lead to more efficient pre-training.

## Installation
* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* **To install fairseq** and develop locally:
``` bash
git clone https://github.com/princeton-nlp/DinkyTrain.git
cd DinkyTrain
pip install --editable ./
```

* **For faster training (FP16)** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

``` bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

* **For faster training (DeepSpeed cuda kernel)** install [DeepSpeed](https://www.deepspeed.ai) library and compile the DeepSpeed kernel

``` bash
DS_BUILD_TRANSFORMER=1 DS_BUILD_STOCHASTIC_TRANSFORMER=1 pip install deepspeed
```

* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`
* If you use Docker make sure to increase the shared memory size either with `--ipc=host` or `--shm-size`
 as command line options to `nvidia-docker run` .

**Trouble-shooting**:
* If using lower version of Python, you might encounter import problems with `importlib.metadata`. Try `pip install importlib-metadata`.
* To install `apex` and `deepspeed`, you will need nvcc (CUDA compiler).
* When installing `apex`, if you encounter the error `Cuda extensions are bing compiled with a version of Cuda that does not match ...`, go to `setup.py` and comment out the line that raised the error (at your own risk).
* Both `apex` and `deepspeed` installation require a high gcc version to support `c++14`. If you encounter relevant errors, update your gcc.

## Data Pre-processing

**Tokenization**:
First, download the GPT2 BPE vocabulary:
```bash
wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
```

Then, tokenize your raw data:
```bash
python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json gpt2_bpe/encoder.json \
    --vocab-bpe gpt2_bpe/vocab.bpe \
    --inputs ${SPLIT}.raw \
    --outputs ${SPLIT}.bpe \
    --keep-empty \
    --workers 8
```

Finally, index and binarize your data:
```bash
fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref ${TRAIN_SPLIT}.bpe \
    --validpref ${VALID_SPLIT}.bpe \
    --testpref ${TEST_SPLIT}.bpe \
    --destdir output-bin \
    --workers 8
```

**Alternatively: Use our pre-processed data**: We preprocessed Wikipedia+BookCorpus and shared it on [Huggingface dataset](https://huggingface.co/datasets/princeton-nlp/wikibook_fairseq_format).
It is ~22GB and contains two epochs of data, each epoch being sliced into 8 shards.
 You can download it using `git`:
```bash
git lfs install # Git lfs is needed for downloading
git clone https://huggingface.co/datasets/princeton-nlp/wikibook_fairseq_format
```

## Pre-training

Use our script for efficient pre-training
``` bash
GPU={number of GPUs} DATA_DIR={data path} [DEEPSPEED=1] bash run_efficient_mlm_recipe.sh
```
Flags explained
* `GPU`: number of GPUs.
* `DATA_DIR`: directory to the processed pre-training data. If you are using our preprocessed dataset, `DATA_DIR` should be:
```bash
DATA_DIR=$(seq 0 15 | sed -e 's/^/wikibook_fairseq_format\/bin-shard/' | sed -e 's/$/-8/' | paste -sd ':')
```
* `DEEPSPEED` (optional): if set to 1, the DeepSpeed CUDA kernel will be used.

Please refer to the script for more hyperparameter choices.

## Fine-tuning on GLUE and SQuAD

All our checkpoints can be converted to HuggingFace [transformers](https://github.com/huggingface/transformers) models (see next section) and use the [transformers](https://github.com/huggingface/transformers) package for fine-tuning. Fairseq also supports fine-tuning on GLUE. 

First, download the preprocessed GLUE data (you can also process by yourself following the preprocess section above):
``` bash
git lfs install # Git lfs is needed for downloading
git clone https://huggingface.co/datasets/princeton-nlp/glue_fairseq_format
```

Then use the following script for fine-tuning
```
DATA_DIR={path to the data directory} \
TASK={glue task name (mnli qnli qqp rte sst2 mrpc cola stsb)} \
LR={learning rate} \
BSZ={batch size} \
EPOCHS={number of epochs} \
SEED={random seed} \
CKPT_DIR={checkpoint's directory} \
CKPT_NAME={checkpoint's name} \
[DEEPSPEED=1] bash finetune_glue.sh
```

For fine-tuning on SQuAD, please convert the models to HuggingFace checkpoints following the next section and use HuggingFace's [examples](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering).

## Convert to HuggingFace

We also provide conversion codes so that you can easily turn Fairseq checkpoints into HuggingFace checkpoints. Usage:

```bash
cd scripts
[PRELAYERNORM=1] [FROM_DS=1] python convert_fs_ckpt_to_hf_ckpt.py --fr {fairseq checkpoint} --to {huggingface checkpoint path} --hf_model_config {roberta-base/roberta-large}
```

Flags explained:
* `PRELAYERNORM=1`: Using pre layer-norm (default is post layer-norm).
* `FROM_DS=1`: The Fairseq checkpoint uses DeepSpeed's cuda kernel.
* `--fr`: The path to the Fairseq checkpoint.
* `--to`: The path you want to save the HuggingFace checkpoint to.
* `--hf_model_config`: `roberta-base` or `roberta-large`.

**IMPORTANT**: all our models use pre layer norm, which is not supported by HuggingFace yet. To use it, import the model class from `huggingface/modeling_roberta_prelayernorm.py`. For example:
``` python
from huggingface.modeling_roberta_prelayernorm import RobertaForSequenceClassification
```

For more configuration, please refer to `convert_fs_ckpt_to_hf_ckpt.py`.

## Model List

Here are the HuggingFace checkpoints of our models in the paper [Should You Mask 15% in Masked Language Modeling](https://arxiv.org/abs/2202.08005). Results are development set performance.
|              Model              | MNLI | QNLI | QQP |  SST-2 
|:-------------------------------|:--------:|:---------:|:---------:|:---------:|
|  [princeton-nlp/efficient_mlm_m0.15](https://huggingface.co/princeton-nlp/efficient_mlm_m0.15) |  84.2 |  90.9 |  87.8 |  93.3 | 
|  [princeton-nlp/efficient_mlm_m0.20](https://huggingface.co/princeton-nlp/efficient_mlm_m0.20) |  84.1 |  91.3 |  87.9 | 92.7 | 
|  [princeton-nlp/efficient_mlm_m0.30](https://huggingface.co/princeton-nlp/efficient_mlm_m0.30) |  84.2 | 91.6  | 88.0 | 93.0  | 
| [princeton-nlp/efficient_mlm_m0.40](https://huggingface.co/princeton-nlp/efficient_mlm_m0.40) |  84.5  | 91.6 | 88.1 | 92.8 |
|  [princeton-nlp/efficient_mlm_m0.50](https://huggingface.co/princeton-nlp/efficient_mlm_m0.50) | 84.1  |  91.1 | 88.1  | 92.7  | 
|  [princeton-nlp/efficient_mlm_m0.60](https://huggingface.co/princeton-nlp/efficient_mlm_m0.60) | 83.2  | 90.7  | 87.8  | 92.6  | 
|  [princeton-nlp/efficient_mlm_m0.70](https://huggingface.co/princeton-nlp/efficient_mlm_m0.70) | 82.3  | 89.4  | 87.5  | 91.9  | 
|  [princeton-nlp/efficient_mlm_m0.80](https://huggingface.co/princeton-nlp/efficient_mlm_m0.80) | 80.8  | 87.9  | 87.1  | 90.5  | 
|  [princeton-nlp/efficient_mlm_m0.15-801010](https://huggingface.co/princeton-nlp/efficient_mlm_m0.15-801010)    |   83.7  | 90.4 | 87.8 |  93.2 | 
|  [princeton-nlp/efficient_mlm_m0.40-801010](https://huggingface.co/princeton-nlp/efficient_mlm_m0.40-801010)   |   84.3  | 91.2 | 87.9 |  93.0 |

We also offer the original (deepspeed) fairseq checkpoints [here](https://huggingface.co/princeton-nlp/efficient_mlm_fairseq_ckpt). 


## Bugs or Questions?
If you have any questions, or encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation

``` bibtex
@article{wettig2022should,
   title={Should You Mask 15% in Masked Language Modeling?},
   author={Wettig, Alexander and Gao, Tianyu and Zhong, Zexuan and Chen, Danqi},
   boo={arXiv preprint arXiv:2202.08005},
   year={2022}
}
```

## Acknowledgment

* Our package is based on [fairseq](https://github.com/pytorch/fairseq):

Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. 2019. fairseq: A fast, extensible toolkit for sequence modeling. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations)_, pages 48–53.

* Our efficient training recipe is based on the following paper:

Peter Izsak, Moshe Berchansky, and Omer Levy. 2021. How to train BERT with an academic budget. In _Empirical Methods in Natural Language Processing (EMNLP)_, pages 10644–10652.
