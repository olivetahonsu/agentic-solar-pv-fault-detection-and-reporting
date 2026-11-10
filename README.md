# agentic-solar-pv-fault-detection-and-reporting
An agentic system that detects solar pv faults, reports and recommend solutions.


# Running the Sanity Check

This validates that the full pipeline (data loading, model, training, evaluation) works correctly on a small 100-image sample, **before** committing to the full training run on the complete dataset.

## 1. Prerequisites

- Python environment with internet access (needed to download the stock `yolo26n.pt` weights on first run)
- Folders `sample_100/` and `sanity_check.py` sitting in the same directory

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

(Installs `ultralytics` and `pyyaml`. This pulls in PyTorch, so the first install may take a few minutes.)

## 3. Run it

```bash
python sanity_check.py
```

In a Jupyter/Colab notebook, use either:

```python
!python sanity_check.py
```

or

```python
%run sanity_check.py
```

No arguments needed — it runs on CPU automatically if no GPU is available.

## 4. What it does

1. Checks PyTorch/Ultralytics are installed and reports whether a GPU is available
2. Builds `sample_100/data.yaml` automatically (generated fresh each run — not something you create manually)
3. Validates all labels in `sample_100/labels/` (class IDs in range, coordinates normalized)
4. Loads the stock, pretrained `yolo26n.pt` model
5. Trains for 3 epochs on the 100-image sample
6. Runs validation and inference checks

## 5. What a pass looks like

- No crashes, no NaN losses
- Ends with: `SANITY CHECK PASSED`
- Precision/recall/mAP values will be **low and not meaningful** — that's expected (3 epochs, 100 images, train=val). This step only confirms the pipeline runs correctly, not that the model is accurate.

## Troubleshooting

**`OMP: Error #15` (Windows)** — add this to the top of your script/notebook cell, before importing torch/ultralytics:

```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

## Next step

Once this passes, proceed to the full training run with `train.py` on the complete dataset (GPU recommended).
