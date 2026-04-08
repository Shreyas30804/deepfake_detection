# deepfake_detection

Starter repository for deepfake detection experiments, notebooks, and model code.

## Project Structure

```text
deepfake_detection/
|-- notebooks/
|   `-- baseline_experiment.ipynb
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data.py
|   |-- model.py
|   `-- train.py
|-- train.py
|-- requirements.txt
`-- README.md
```

## Quick Start

1. Create a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Point `TrainingConfig.data_dir` at your DFDC-style dataset.
4. Run `python train.py`.

## Colab Workflow

Clone the repository in a notebook:

```python
!git clone https://github.com/Shreyas30804/deepfake_detection.git
%cd deepfake_detection
!pip install -r requirements-colab.txt
```

Then import and reuse the training code from `src/`.

Use `requirements.txt` for a fresh local environment and `requirements-colab.txt` for Google Colab, since Colab already includes its own PyTorch and Pillow stack.

## What Changed

- Training and preprocessing logic now lives in `src/` instead of a large notebook.
- The dataset is split at the video level before face extraction, which avoids train/validation leakage.
- Face crops are taken from the original frame using MTCNN detection boxes rather than converting the detector tensor output directly.
- Gradient accumulation now applies the final partial step instead of silently dropping it.

## Notes

- Keep heavy datasets and checkpoints outside Git history.
- Store reusable logic in `src/` and keep notebooks focused on experiments.
- Replace the placeholder model and dataset logic with your real pipeline.
