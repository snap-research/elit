# [CVPR 2026] Elastic Latent Interfaces

[One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers](TODO)


[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://snap-research.github.io/elit/)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b)](https://arxiv.org/abs/XXXX.XXXXX)
<!-- [![GitHub](https://img.shields.io/badge/GitHub-Code-black.svg)](https://github.com/snap-research/elit) -->

**Moayed Haji-Ali<sup>1,2</sup>, Willi Menapace<sup>2</sup>, Ivan Skorokhodov<sup>2</sup>, Dogyun Park<sup>2</sup>, Anil Kag<sup>2</sup>, Michael Vasilkovsky<sup>2</sup>, Sergey Tulyakov<sup>2</sup>, Vicente Ordonez<sup>1</sup>, Aliaksandr Siarohin<sup>2</sup>**

*<sup>1</sup>Rice University, <sup>2</sup>Snap Inc.

## 🚀 Check Out Our Latest Work! 🎥🔊 

> Our other work **[DFM: Decomposable Flow Matching](https://snap-research.github.io/dfm/)** — a simple framework for progressive scale-by-scale generation that achieves up to **50% faster convergence** compared to Flow Matching. **This repo also contains the code for DFM.**

---

<img src="assets/teaser.png" width="1200">

## TL;DR

> We found that DiTs waste substantial compute by allocating it uniformly across pixels, despite large variation in regional difficulty. **ELIT** addresses this by introducing a variable-length set of *latent tokens* and two lightweight cross-attention layers (Read & Write) that concentrate computation on the most important input regions, delivering up to **53% FID** and **58% FDD improvements** on ImageNet-1K 512px. At inference time, the number of latent tokens becomes a user-controlled knob, providing a smooth **quality–FLOPs trade-off**  while enabling **~33% cheaper guidance**  out of the box.

---

## Method Implementation

ELIT introduces a minimal change to DiT-like architectures: a **latent interface** — a variable-length token sequence — coupled with lightweight **Read** and **Write** cross-attention layers.

1. A **latent interface** of *K* tokens is instantiated.
2. A lightweight **Read** cross-attention layer pulls information from spatial tokens into the latent interface, prioritizing harder regions using grouped cross-attention.
3. Standard transformer blocks operate on the latent tokens.
4. A **Write** cross-attention layer maps the latent updates back to the spatial grid.
5. During training, tail latents are randomly dropped, making the latent interface importance-ordered.
6. At inference, the number of latents serves as a user-controlled compute knob.

---

## Disclaimer

This repo provides a reimplementation of ELIT on top of SiT, following [REPA](https://github.com/sihyun-yu/REPA) setup. The architecture does not exactly follow the one used in the paper and results might be different. Below, we provide comparison between SiT and ELIT produced using this repo.

---

## Experimental Results

### ImageNet 256×256

| Method | FID↓ | sFID↓ | IS↑ | Precision↑ | Recall↑ | FLOPs |
|--------|------|-------|-----|------------|---------|-------|
| SiT-XL/2 | 18.58 | 5.38 | 75.39 | 0.246 | 0.526 | 182 |
| ELIT-SiT-XL/2 | 11.74 | 6.03 | 112.11 | 0.315 | 0.549 | 188 |
| ELIT-SiT-XL/2 (multibudget) | 12.61 | 6.25 | 110.11 | 0.305 | 0.558 | 127 |

### ImageNet 512×512

| Method | FID↓ | sFID↓ | IS↑ | Precision↑ | Recall↑ | FLOPs |
|--------|------|-------|-----|------------|---------|-------|
| SiT-XL/2 | 33.24 | 8.55 | 48.22 | 0.308 | 0.581 | 806 |
| ELIT-SiT-XL/2 | 10.82 | 6.53 | 114.82 | 0.489 | 0.522 | 831 |
| ELIT-SiT-XL/2 (multibudget) | 9.55 | 6.65 | 122.56 | 0.497 | 0.531 | 536 |



<!-- TODO: Add pretrained checkpoints and fill in FLOPs -->
Pretrained checkpoints of the above experiments will be released soon.

---

## 1. Environment Setup

```bash
conda create -n elit python=3.9 -y
conda activate elit
pip install -r requirements.txt
```

---

## 2. Dataset

### 2.1 Dataset Download

Download [ImageNet](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data). Then run the following processing and VAE latent extraction scripts.

```bash
# Convert raw ImageNet data to a ZIP archive at 256x256 resolution
python dataset_tools.py convert \
    --source=[YOUR_DOWNLOAD_PATH]/ILSVRC/Data/CLS-LOC/train \
    --dest=[TARGET_PATH]/images \
    --resolution=256x256 \
    --transform=center-crop-dhariwal
```

```bash
# Convert the pixel data to VAE latents
python dataset_tools.py encode \
    --source=[TARGET_PATH]/images \
    --dest=[TARGET_PATH]/vae-sd
```

Here, `YOUR_DOWNLOAD_PATH` is the directory where you downloaded the dataset, and `TARGET_PATH` is the directory where you will save the preprocessed images and corresponding compressed latent vectors. This directory will be used for your experiment scripts.

---

## 3. Training

Training uses the unified `train.py` script with YAML configuration files or CLI arguments. Update `data_dir` in the config to point to your data directory.


```bash
# From CLI args
accelerate launch train.py --model [MODEL_NAME] --exp-name [EXP_NAME] --data-dir [DATA_DIR]

# Or from YAML config
accelerate launch train.py --config [CONFIG_PATH] --data-dir [DATA_DIR]
```

where [MODEL_NAME] can be specificed as SiT or ELIT-SiT baselines (e.g SiT-XL/2 or ELIT-SiT-XL/2)

Sample training configurations can be found in `experiments/train`


### Example Training 

```bash
# From CLI args
accelerate launch train.py --model ELIT-SiT-XL/2 --exp-name elit-sit-xl-2-256px --data-dir [DATA_DIR]

# Or from YAML config
accelerate launch train.py --config experiments_updated/train/elit_sit_b_256.yaml --data-dir [DATA_DIR]
```



### Key ELIT Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | Model architecture: `ELIT-SiT-B/2`, `ELIT-SiT-L/2`, `ELIT-SiT-XL/2` | — |
| `elit_max_mask_prob` | Maximum masking probability for tail-dropping during training. | `0.0` |
| `elit_min_mask_prob` | Minimum masking probability. Defaults to `elit_max_mask_prob` (single budget). When different from max, mask probability is uniformly sampled from valid levels in `[min, max]`. | `None` (= max) |
| `elit_group_size` | Group size for grouped cross-attention in Read/Write layers. We recommend 4 for 256px and 8 for 512px, resulting in 16 groups | `4` |


### Multibudget training 
```bash

# 256px — sample all valid budgets (min=0, max not set → defaults to 1-1/16=0.9375 for group_size=4)
accelerate launch train.py --model ELIT-SiT-XL/2 --exp-name elit-sit-xl-2-256px --data-dir [DATA_DIR] --elit-min-mask-prob 0 --elit-max-mask-prob 0.9375 --elit_group_size 4

# 512px — sample all valid budgets
accelerate launch train.py --model ELIT-SiT-XL/2 --exp-name elit-sit-xl-2-512px --data-dir [DATA_DIR] --elit-min-mask-prob 0 --elit-max-mask-prob 0.9375 --elit_group_size 8

# 256px — sample budgets between 50% and 75% masking
accelerate launch train.py --model ELIT-SiT-XL/2 --exp-name elit-sit-xl-2-256px --data-dir [DATA_DIR] --elit-min-mask-prob 0.5 --elit-max-mask-prob 0.75 --elit_group_size 4
```

### DFM training
This repo also support training for [Decomposable Flow Matching (DFM)](https://snap-research.github.io/dfm/). Yoy can enable training by choosing the DFM model family (e.g `DFM-SiT-XL/2`,
`DFM-SiT-B/2`, etc).

```bash
accelerate launch train.py --model DFM-SiT-XL/2 --exp-name dfm-sit-xl-2-256px --data-dir [DATA_DIR]
```
Please refer to [DFM repo](https://github.com/snap-research/dfm) for full details on hyperparameters.

---

## 4. Sampling

Sampling uses the unified `generate.py` script with DDP:

### 4.1 ELIT-SiT

```bash
# From CLI args
torchrun --nproc_per_node=8 generate.py \
    --model ELIT-SiT-XL/2 --ckpt exps/elit-sit-xl-2-256px/checkpoints/0400000.pt

# Or from YAML config
torchrun --nproc_per_node=8 generate.py \
    --config experiments_updated/generation/elit_sit_b_256.yaml \
    --ckpt exps/elit-sit-b-2-256px/checkpoints/0400000.pt
```

### 4.2 Variable Budget Inference

ELIT supports controlling the inference budget via the `--inference-budget` argument. This specifies the fraction of latent tokens to use:

```bash
# Full budget (100% tokens)
torchrun --nproc_per_node=8 generate.py \
    --config experiments_updated/generation/elit_sit_b_256.yaml \
    --ckpt path/to/ckpt.pt --inference-budget 1.0

# Half budget (50% tokens) — ~50% fewer FLOPs in the core transformer
torchrun --nproc_per_node=8 generate.py \
    --config experiments_updated/generation/elit_sit_b_256.yaml \
    --ckpt path/to/ckpt.pt --inference-budget 0.5

# Quarter budget (25% tokens)
torchrun --nproc_per_node=8 generate.py \
    --config experiments_updated/generation/elit_sit_b_256.yaml \
    --ckpt path/to/ckpt.pt --inference-budget 0.25
```

### 4.3 Multi-Budget Analysis

To generate images at **all** budgets, measure latency and FLOPs, and produce comparison plots:

```bash
python elit_multibudget_inference.py \
    --model ELIT-SiT-XL/2 \
    --ckpt path/to/ckpt.pt \
    --resolution 256 \
    --class-label 207 \
    --output-dir multibudget_results
```

---

## 5. Evaluation

We provide evaluation scripts in `experiments_updated/evaluation/` that generate samples and compute FID, sFID, IS, Precision, and Recall.

```bash
bash experiments_updated/evaluation/eval_elit_sit_b_256.sh
```

This will generate samples under the `results/` directory and an `.npz` file which can be used for evaluation. To run the reference TensorFlow evaluation on ImageNet, we use the [ADM evaluation](https://github.com/openai/guided-diffusion/tree/main/evaluations) suite.

---

<!-- ## Pretrained Checkpoints -->

<!-- TODO: Add pretrained checkpoint download links -->
<!-- TODO: Add checkpoint table with model name, resolution, FID, download link -->

<!-- | Model | Resolution | Steps | FID | sFID | IS | Download |
|-------|-----------|-------|-----|------|----|----------|
| ELIT-SiT-B/2 | 256×256 | 400K | — | — | — | TODO |
| ELIT-SiT-XL/2 | 256×256 | 400K | — | — | — | TODO |
| ELIT-SiT-XL/2 | 512×512 | 400K | — | — | — | TODO |

--- -->

## Large-scale training strategy
For large-scale training, we recommend using the settings in Appendix D: increase model capacity while keeping compute bounded by  reducing tokens at the bottleneck. Concretely, we drop75% of tokens in the bottleneck throughout training, so the model can prioritize learning global structure while still benefiting from a larger parameter budget without increasing training or inference FLOPs.

```bash
# single budget
accelerate launch train.py --model ELIT-SiT-XL/2 --exp-name elit-sit-xl-2-256px --data-dir [DATA_DIR] --elit-max-mask-prob 0.75 --elit_group_size 4

# multibudget
accelerate launch train.py --model ELIT-SiT-XL/2 --exp-name elit-sit-xl-2-256px --data-dir [DATA_DIR] --elit-min-mask-prob 0.75 --elit_group_size 4 --elit-max-mask-prob 0.9375 --elit_group_size 4

```


## Acknowledgement

This code is mainly built upon [REPA](https://github.com/sihyun-yu/REPA). We thank the authors for open-sourcing their codebase.

---

## BibTeX

```bibtex
@inproceedings{hajiali2026elit,
  title={One Model, Many Budgets: Elastic Latent Interfaces
    for Diffusion Transformers},
  author={Moayed Haji-Ali and Willi Menapace and Ivan Skorokhodov
    and Dogyun Park and Anil Kag and Michael Vasilkovsky
    and Sergey Tulyakov and Vicente Ordonez
    and Aliaksandr Siarohin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer
    Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
