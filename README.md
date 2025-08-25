# PDC Project — ML Pipeline for Binary Classification


## What this project contains
- Notebooks: `claude.ipynb`, `implementation code.ipynb` — main pipeline implementations and experiments.
- Scripts: `serial.py`, `parallel.py`, `gpu.py` — helper/alternate implementations.
- Dataset: `pdc_dataset_with_target.csv` (placed in the project root).



## Prerequisites
- OS: Windows (tested environment for this repo is Windows).
- Python: use Python 3.8–3.10 (recommended). Newer Python versions (3.11+) may cause binary issues with TensorFlow on Windows.
- GPU: optional. If you plan to use GPU-accelerated TensorFlow, ensure CUDA/cuDNN and GPU drivers match the TensorFlow release. PyTorch is used in parts of the project and is generally easier to install for GPU support.

Recommended packages (minimum):
- numpy, pandas, scikit-learn, matplotlib, seaborn
- dask, joblib
- torch (PyTorch)
- tensorflow (optional; may require specific Python/CUDA versions)

## Quick setup (PowerShell)
Create and activate a virtual environment, upgrade pip, and install required packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# PDC Project — ML pipeline for binary classification

A small research / course project implementing an end-to-end machine learning pipeline with serial, parallel and GPU-accelerated training/experiments. The code and notebooks demonstrate data loading, preprocessing, model training (PyTorch + scikit-learn), and simple performance comparisons.

This README is written to be GitHub-friendly and generic so you can upload the repository as-is.

## Repository structure

- `claude.ipynb` — primary pipeline notebook (data loading, preprocessing, training modes).
- `implementation code.ipynb` — additional experiments and implementation notes.
- `serial.py`, `parallel.py`, `gpu.py` — script variants / helpers (if present in the repo).
- `pdc_dataset_with_target.csv` — dataset (place in repository root or provide a full path).
- Slides / reports: presentation and performance report files used for the course.

Note: file names may vary slightly; open the project root to verify exact filenames.

## Quick start (recommended)

1. Clone or copy this repository to your machine.
2. Place the dataset `pdc_dataset_with_target.csv` in the repository root, or note its absolute path.
3. Create and activate a Python virtual environment and install dependencies.

Example (PowerShell):

```powershell
python -m venv .venv
.
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you don't have a `requirements.txt` file, install the essentials:

```powershell
pip install numpy pandas scikit-learn matplotlib seaborn dask joblib torch
# Optional: TensorFlow (only if compatible with your Python/CUDA setup)
# pip install tensorflow==2.10.0
```

## Usage

Open the notebooks in Jupyter for step-by-step execution (recommended for exploration):

```powershell
jupyter notebook
```

The primary notebook is `claude.ipynb`. It accepts a `--data` CLI argument when run as a script, but also has a default dataset path so it can be executed interactively inside Jupyter without providing CLI arguments.

Run the notebook as a script (optional):

```powershell
jupyter nbconvert --to script claude.ipynb
python claude.py --data "pdc_dataset_with_target.csv" --all
```

Or run the helper scripts (if included):

```powershell
python serial.py
python parallel.py --data "pdc_dataset_with_target.csv"
python gpu.py
```

## Configuration

- Python version: 3.8–3.10 recommended for best compatibility with TensorFlow on Windows.
- GPU: PyTorch is used for GPU-accelerated training paths. If you plan to use TensorFlow with GPU, follow TensorFlow's GPU installation guide and match CUDA/cuDNN versions exactly to the TensorFlow release.

## Troubleshooting

- DLL load failed when importing TensorFlow:
  - Use Python 3.8–3.10 for stable TensorFlow Windows wheels.
  - Install the Microsoft Visual C++ Redistributable (latest supported).
  - If using GPU TensorFlow, ensure CUDA and cuDNN match the TensorFlow release.
  - As a workaround, use the PyTorch training code included in the notebooks.

- `argparse` errors when running inside notebooks:
  - Notebooks don't receive CLI arguments by default. The primary notebook includes a default path so it runs interactively. When converting to a script, provide `--data` if your dataset is outside the repo root.

