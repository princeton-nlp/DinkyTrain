<p align="center">
  <a href="https://github.com/fairseq/fairseq"><img src="docs/pnlp_logo512.png" width="150" style="padding-right: 2em"></a>
  <a href="https://princeton-nlp.github.io/"><img src="docs/fairseq_logo.png" width="150"></a>
  <br />
</p>

This repository is a fork of [fairseq](https://github.com/fairseq/fairseq) with custom changes for efficient training of masked language models (MLM). We fork fairseq to give researchers more flexibility when using our training scripts,
while also making it easier to adapt our code contributions into other projects.

Our contributions:
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
...
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
...
### Pre-training

Use our script for efficient pre-training
``` bash
GPU={number of GPUs} DATA_DIR={data path} [DEEPSPEED=1] bash run_efficient_mlm_recipe.sh
```
Flags explained
* `GPU`: number of GPUs.
* `DATA_DIR`: directory to the processed pre-training data.
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

## Citations