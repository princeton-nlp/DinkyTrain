<p align="center">
  <!--<a href="https://github.com/fairseq/fairseq"><img src="docs/pnlp_logo512.png" width="150"></a>-->
  <a href="https://github.com/princeton-nlp/DinkyTrain"><img src="docs/DinkyTrainLogo.png" width="200"></a>
  <!--<a href="https://princeton-nlp.github.io/"><img src="docs/fairseq_logo.png" width="150"></a>-->
  <br />
</p>

This repository is a library for efficient training of masked language models (MLM), built with [fairseq](https://github.com/fairseq/fairseq). We fork fairseq to give researchers more flexibility when using our training scripts,
while also making it easier to adapt our code contributions into other projects.

## Why DinkyTrain?
The [Dinky](https://en.wikipedia.org/wiki/Princeton_Branch) runs between Princeton Junction and Princeton and is the shortest scheduled commuter rail line in the United States.
We also aim to make pre-training short and accessible to everyone.

## Our contributions
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



# Efficient MLM Pre-training
## Overview
You can reproduce the pre-training experiments of our recent paper [Should You Mask 15% in Masked Language Modeling?](https://arxiv.org/abs/2202.08005),
where we find that higher masking rates can lead to more efficient pre-training.
Citation:
```
@article{wettig2022should,
  title={Should You Mask 15\% in Masked Language Modeling?},
  author={Wettig, Alexander and Gao, Tianyu and Zhong, Zexuan and Chen, Danqi},
  journal={arXiv preprint arXiv:2202.08005},
  year={2022}
}
```

## Installation
* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* **To install fairseq** and develop locally:
``` bash
git clone https://github.com/pytorch/fairseq
cd fairseq
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

## Run the Pre-training
### Data Pre-processing

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

### Pre-training

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
* `DEEPSPEED`: if set to 1, the DeepSpeed CUDA kernel will be used.

Please refer to the script for more hyperparameter choices.

### Fine-tuning
...
### Convert to HuggingFace

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

For more configuration, please refer to `convert_fs_ckpt_to_hf_ckpt.py`.

## Model List
...




<!-- # Citation

Please cite as:

``` bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
``` -->
