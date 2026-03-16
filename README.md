# DWMamba

Welcome to the official repository for **DWMamba**. Our codebase is specifically designed for efficient image super-resolution utilizing the Mamba architecture.

## Prerequisites

Our codebase was successfully tested with the following environment configurations:
* **OS:** Ubuntu 20.04
* **CUDA:** 11.7
* **Python:** 3.9
* **PyTorch:** 2.0.1 + cu117

---

## Installation Guide

The following steps will help you set up the build environment from scratch.

### Step 1: Create and activate a virtual environment
```bash
conda create -n dwmamba python=3.9 -y
conda activate dwmamba
```

### Step 2: Install PyTorch Core Framework
> **Note:** In order to prevent the latest version of MKL from causing `iJIT_NotifyEvent` errors in PyTorch 2.0.1, you must force the installation of an older version of `mkl` before installing PyTorch.

```bash
conda install "mkl<2024.1" -y
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

### Step 3: Install dependency packages
```bash
pip install -r requirements.txt
```

### Step 4: Compile and install Mamba core operators
> **Important:** Please ensure PyTorch is fully installed before running this step. Compiling CUDA operators may take a few minutes depending on your hardware.

```bash
pip install causal-conv1d==1.0.0
pip install mamba-ssm==1.0.1
```

---

## Testing / Evaluation

To run the evaluation, execute the following command. Make sure to replace `/path/to/your/dataset` with the actual path to your downloaded dataset.

```bash
CUDA_VISIBLE_DEVICES=0 python test_demo.py \
    --data_dir /path/to/your/dataset \
    --save_dir ./results \
    --model_id 12
```


## Efficiency Measurement

To quickly evaluate the computational costs of the model without loading any image datasets, we provide a standalone script. It measures **Parameters, FLOPs, Activations, and Max Memory** using a dummy input tensor (default shape `1x3x256x256`).

Execute the following command:

```bash
python quick_test.py
```

The script will print the metrics to the console and automatically save a detailed summary in `model_efficiency_report.txt`.