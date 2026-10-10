# ConvNeXt on CIFAR-10 — AI Assignment #2 (CISC3024 Pattern Recognition)

Re-implementation of **ConvNeXt** (*A ConvNet for the 2020s*, CVPR 2022, arXiv:2201.03545 — a
**Deep CNN**) for **CIFAR-10 image classification**, produced end-to-end with an AI coding agent
under the assignment rule that *no code may be written by hand*.

## Headline result

| Run | Setting | Test Top-1 |
|---|---|---|
| `full_convnext_tiny_cifar10` | ConvNeXt-Tiny, 60 epochs, bfloat16, single 8 GB GPU | **92.66 %** (92.87 % with test-time augmentation) |
| `mvp_atto_subset5k` | atto preset, 4,500 images, CPU, 10 epochs (pipeline validation) | 45.05 % |
| `full_convnext_tiny_cifar10_amp_fp16_underflow` | same recipe under fp16 — **failed run, kept as a controlled comparison** | 25.73 % |

Best validation accuracy was 93.50 % (epoch 59); the model has 27.8 M parameters. The failed fp16
run is retained on purpose: its checkpoints made it possible to diagnose the collapse of the
gradient-scaler loss scale, which is discussed in the report (Section 4.6).

## Repository layout

| Folder | Contents |
|---|---|
| `01_requirements/` | Assignment statement and requirement analysis |
| `02_ai_chats/` | AI dialogue records, candidate list and the algorithm profile |
| `03_code/` | The complete PyTorch project (model, training, evaluation, utilities, downloader) and the self-checking Colab notebook generator |
| `04_data/` | Dataset description, split index files and download instructions |
| `05_results/` | Every experiment with its figures, logs, metrics and README |
| `06_report/` | The final report, in Word and PDF |

## How to run

```bash
# environment (PyPI's Windows wheels are CPU-only; use the CUDA index for GPU work)
pip install --upgrade --extra-index-url https://download.pytorch.org/whl/cu130 \
            "torch==2.14.1+cu130" "torchvision==0.29.1+cu130"
pip install -r 03_code/requirements.txt

# full-scale training (ConvNeXt-Tiny, 60 epochs, bfloat16, one GPU)
cd 03_code
python train.py --run-tag full_convnext_tiny_cifar10 --model tiny --epochs 60 \
    --batch-size 128 --stem-mode conv --device cuda --num-workers 0 \
    --data-root ../04_data/cifar10 --output-dir ../05_results

# evaluation with test-time augmentation
python evaluate.py --ckpt ../05_results/full_convnext_tiny_cifar10/models/best.pt \
    --device cuda --tta --data-root ../04_data/cifar10 --output-dir ../05_results
```

A self-contained Colab notebook (`03_code/notebooks/ConvNeXt_CIFAR10_Colab.ipynb`) rebuilds the
whole project inside Colab, downloads CIFAR-10, runs the training and displays the figures, so the
experiments can be reproduced with no local setup.

## Not committed (size)

The raw CIFAR-10 archive (~341 MB), the model checkpoints (`*.pt`, ~716 MB) and the packaged
results archive (`*.zip`, ~271 MB) are excluded by `.gitignore`. The dataset is downloadable from
the official source described in `04_data/dataset_info.md`; every reported metric, figure and log
needed to verify the results is included in `05_results/`.

## Reproducibility

Every run is identified by a `--run-tag`, writes its complete hyper-parameter set to
`metrics/config.json`, appends per-epoch history after each epoch, and stores the optimiser,
scheduler and scaler state inside its checkpoints. Seeds for Python, NumPy, PyTorch and the loader
workers are fixed, and the train/validation split is derived from a fixed-seed permutation that is
archived as index files.

## AI assistance

All code, experiments and report text were produced by AI tools (an AI coding agent and large
language models), as required by the assignment; nothing was written by hand. The dialogue records
documenting this process are in `02_ai_chats/`.
