<p align="center">
  <a href="https://github.com/fairseq/fairseq"><img src="docs/pnlp_logo512.png" width="150" style="padding-right: 2em"></a>
  <a href="https://princeton-nlp.github.io/"><img src="docs/fairseq_logo.png" width="150"></a>
  <br />
</p>

This repository is a fork of [fairseq](https://github.com/fairseq/fairseq) with custom changes for efficient training of masked language models. We fork fairseq to give researchers more flexibility when using our training scripts,
while also making it easier to adapt our code contributions into other projects.

Our contributions:
* [DeepSpeed transformer kernel](https://www.deepspeed.ai/tutorials/transformer_kernel/) integration
* A training recipe for efficient MLM pre-training (soon)


Other [fairseq features](https://github.com/fairseq/fairseq#features):
* multi-GPU training on one machine or across multiple machines (data and model parallel)
* fast generation on both CPU and GPU with multiple search algorithms implemented:
  + beam search
  + Diverse Beam Search ([Vijayakumar et al., 2016](https://arxiv.org/abs/1610.02424))
  + sampling (unconstrained, top-k and top-p/nucleus)
  + [lexically constrained decoding](examples/constrained_decoding/README.md) (Post & Vilar, 2018)
* [gradient accumulation](https://fairseq.readthedocs.io/en/latest/getting_started.html#large-mini-batch-training-with-delayed-updates) enables training with large mini-batches even on a single GPU
* [mixed precision training](https://fairseq.readthedocs.io/en/latest/getting_started.html#training-with-half-precision-floating-point-fp16) (trains faster with less GPU memory on [NVIDIA tensor cores](https://developer.nvidia.com/tensor-cores))
* [extensible](https://fairseq.readthedocs.io/en/latest/overview.html): easily register new models, criterions, tasks, optimizers and learning rate schedulers
* [flexible configuration](docs/hydra_integration.md) based on [Hydra](https://github.com/facebookresearch/hydra) allowing a combination of code, command-line and file based configuration
* [full parameter and optimizer state sharding](examples/fully_sharded_data_parallel/README.md)
* [offloading parameters to CPU](examples/fully_sharded_data_parallel/README.md)

See the [fairseq repo](https://github.com/fairseq/fairseq) and its [documentation](https://fairseq.readthedocs.io/) for more details on how to use and extend fairseq.

# Efficient MLM Pre-training
## Overview
...
## Installation
* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq** and develop locally:

``` bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./

# to install the latest stable release (0.10.x)
# pip install fairseq
```

* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

``` bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`
* If you use Docker make sure to increase the shared memory size either with `--ipc=host` or `--shm-size`
 as command line options to `nvidia-docker run` .

## Run the Pre-training
### Data Pre-processing
...
### Pre-training
...
### Fine-tuning
...
### Convert to HuggingFace
...

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
