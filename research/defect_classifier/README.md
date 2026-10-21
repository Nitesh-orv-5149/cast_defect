# Close-up defect-type classifier (tab 1)

Photograph a defect close up and the app names its type: gas porosity / blowhole, shrinkage cavity, crack,
misrun / cold shut, flash / burr, mould or sand defect, scratch, or good surface. The model is
`web/types.onnx` with `web/types.json`: EfficientNet-B0 with grayscale input at 128 px, 16 MB, about 0.15 s per
photo in the browser. Tab 1 keeps the old whole-impeller defect/OK model behind the "Whole impeller" switch.

## Data

The training set is real photos with a box around every defect, from Roboflow Universe, all CC BY 4.0. Each box
becomes one close-up crop (`build_crops.py`).

| Project | Original photos | Classes used |
|---|---|---|
| [detect-casting-defect/casing_defect_detection](https://universe.roboflow.com/detect-casting-defect/casing_defect_detection) | 1,004 | BlowHoles, Shrinkage, Scratch, MouldDefect, Burr |
| [detect-casting-defect/casting_dataset_kaggle](https://universe.roboflow.com/detect-casting-defect/casting_dataset_kaggle) | 781 | Blowholes, Shrinkage, crack, Misrun, Cold shot, Flash, Scar, scratch |
| [casting-detect/casting-defects-onf6n-xt28x](https://universe.roboflow.com/casting-detect/casting-defects-onf6n-xt28x) | 212 | blow hole, crack, burr, scratch |
| [detect-casting-defect/casting-dataset-2](https://universe.roboflow.com/detect-casting-defect/casting-dataset-2) | 69 | Shrinkage, Porosity |

The good-surface crops come from the defect-free photos of the Kaggle dataset
[real-life-industrial-dataset-of-casting-product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
(ok_front, 519 photos). A few more come from real sand castings in
[SIFA](https://www.kaggle.com/datasets/martellbell17/sandcast-images-for-foundry-automation-sifa).

Notes on the data:
- Augmented copies and photos shared between projects are de-duplicated. The train/val/test split (70/15/15) is
  by source photo, so no photo is on both sides.
- `raw_data_casting` is left out because its boxes don't line up with the defects.
- The Kaggle "casting-defect-detection" set with 12 classes is synthetic: shapes painted onto one impeller photo.
  It is not used.

| Class | Train | Val | Test |
|---|---|---|---|
| flash / burr | 2,318 | 556 | 599 |
| good | 1,718 | 375 | 289 |
| gas | 810 | 208 | 177 |
| shrinkage | 741 | 149 | 159 |
| scratch | 728 | 156 | 184 |
| mould / surface | 427 | 90 | 80 |
| crack | 123 | 32 | 26 |
| misrun | 119 | 26 | 20 |

## Results (held-out test photos)

Accuracy 90.4%, macro-F1 0.83. Defect vs good: 8 of 1,245 defects missed (0.6%), 4 of 289 good surfaces flagged
(1.4%).

| Class | Recall | Precision |
|---|---|---|
| flash / burr | 0.99 | 0.99 |
| good | 0.99 | 0.98 |
| misrun | 0.90 | 0.69 |
| gas | 0.84 | 0.85 |
| mould / surface | 0.78 | 0.86 |
| crack | 0.77 | 0.61 |
| shrinkage | 0.77 | 0.70 |
| scratch | 0.73 | 0.84 |

The main confusion is gas vs shrinkage, both cavities: 20 gas crops were called shrinkage and 21 shrinkage crops
were called gas. The annotators of the source sets are not always consistent there either. Scratch also mixes
with shrinkage and mould marks.

## Limits

- Every training photo is a grayscale top view of one product family, the Kaggle pump impeller, at 300–640 px.
  A typical defect is about 50 px across. Phone close-ups of other castings will look different, so treat the
  type as "likely" until it has been checked on your own parts.
- Adding 20–30 of your own labelled close-ups per class and retraining is the most effective improvement.
- The classifier assumes the defect fills the frame. A whole-part photo in close-up mode still gets a type guess.

## Reproduce (CPU only, about 50 minutes)

    # Kaggle and Roboflow downloads need KAGGLE_USERNAME/KAGGLE_KEY and ROBOFLOW_API_KEY
    pip install torch torchvision timm onnx onnxruntime pillow pyyaml
    python build_crops.py <datasets dir> <crops dir>
    python train_types.py <crops dir> --epochs 12 --size 128 --out ../../web
