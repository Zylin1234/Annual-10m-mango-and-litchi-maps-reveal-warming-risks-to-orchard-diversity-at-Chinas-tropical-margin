# AEF-OCE-Net: model, validation results, and usage

AEF-OCE-Net identifies background, mango, and litchi from precomputed annual
AlphaEarth Foundations (AEF) embeddings. This folder contains the refined PyTorch
implementation; the default complete variant is `C3_hierarchical`.

The documentation uses **litchi**; some source-code identifiers and existing map
filenames use **lychee**. Both names refer to the same crop. Existing identifiers
are retained for compatibility.

The internal validation, comparison, ablation, and computational benchmark
results are presented below with their evaluation protocols.
The machine-readable comparison is [model_accuracy.csv](model_accuracy.csv).

## Internal validation results

| Metric | AEF-OCE-Net |
| --- | ---: |
| Overall accuracy (OA) | **0.9626 (96.26%)** |
| Cohen's kappa | **0.8850** |
| Mean intersection-over-union (mIoU) | **0.8750** |
| Mango F1 | **0.8828** |
| Litchi F1 | **0.9362** |

### Evaluation protocol and interpretation

The reference labels comprise 26 plots, each 10 km x 10 km, across 18
counties and cities in Hainan. The fixed allocation uses **20 training plots and
six validation plots**, with 2025 AEF inputs and three reference classes.
Tiling at 128 x 128 pixels with a 64-pixel stride produces **5,752 training tiles
and 409 validation tiles**.

Segmentation methods are trained and evaluated on image tiles. The four pixel
classifiers use class-stratified pixel samples from the same plot allocation;
their validation pixels are sampled separately from the validation plots.
The pixel-classifier training set contains 115,950 pixels: 46,000 background,
45,000 mango, and 24,950 litchi. Sampling limits are 5,000 target and 1,500 local
background pixels per orchard plot, 5,000 pixels per background plot, and at most
60,000 pixels per class; class weights are `N / (3 * n_c)`.

These internal validation results use a **single fixed split**. Counties occur
in both the training and validation sets. The split is therefore neither
three-fold cross-validation nor county-held-out validation. The three-fold
county-held-out linear probes are separate feature diagnostics; their scores
do not replace the fixed-split comparison results below.

Annual map accuracy is assessed separately using year-specific point samples.
See the
[2017-2025 map accuracy results](../spatial_distribution_maps/README.md).
In particular, **the internal 2025 OA of 96.26% and the annual-map OA measure
different evaluation sets** and must be reported with their respective protocols.

### Comparison and ablation results

OA, kappa, mIoU, and F1 are on a 0-1 scale. Inference entries are mean +/-
standard deviation per 128 x 128-pixel tile; parameters are in millions.

| Model | Parameters (M) | Training time (s) | Inference time (ms/tile) | OA | Kappa | mIoU | Mango F1 | Litchi F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **AEF-OCE-Net** | 0.150 | 36.30 | 1.969 +/- 0.240 | **0.9626** | **0.8850** | **0.8750** | 0.8828 | **0.9362** |
| AEF-OCE-Net without adapter | 0.148 | 35.31 | 0.882 +/- 0.068 | 0.9616 | 0.8806 | 0.8707 | 0.8765 | 0.9350 |
| AEF-OCE-Net without OCE | 0.132 | 33.68 | 1.122 +/- 0.119 | 0.9609 | 0.8778 | 0.8662 | 0.8759 | 0.9281 |
| U-Net | 7.781 | 65.09 | 2.593 +/- 0.174 | 0.9534 | 0.8580 | 0.8483 | 0.8578 | 0.9189 |
| DeepLabV3+ | 3.852 | 65.64 | 1.622 +/- 0.176 | 0.9503 | 0.8493 | 0.8356 | 0.8467 | 0.9078 |
| PSPNet | 3.031 | 63.56 | 1.575 +/- 0.144 | 0.9289 | 0.7856 | 0.7747 | 0.7957 | 0.8557 |
| LightMLP | 0.0168 | 18.53 | 9.791 +/- 1.113 | 0.8097 | 0.7020 | 0.6938 | 0.9138 | 0.7205 |
| Linear probe | 0.000195 | 2.91 | 3.407 +/- 0.122 | 0.8015 | 0.6916 | 0.6826 | 0.9002 | 0.7316 |
| Random forest | Not available | 7.39 | 67.432 +/- 12.867 | 0.8289 | 0.7336 | 0.7173 | 0.9036 | 0.7759 |
| XGBoost | Not available | 7.59 | 36.397 +/- 5.357 | 0.8317 | 0.7369 | 0.7223 | 0.9151 | 0.7651 |

Computational benchmarks used an NVIDIA GeForce RTX 5080 Laptop GPU with 16 GB
VRAM, an Intel Core Ultra 9 275HX processor, and 32 GB RAM. Training time means
**seconds per epoch** for segmentation models and AEF-OCE-Net variants, and
**seconds per complete fit** for the four pixel classifiers. Inference used
GPU FP32 for segmentation methods and AEF-OCE-Net variants, and the CPU for
LightMLP, the linear probe, random forest, and XGBoost. The time columns must
therefore be interpreted with their training units and inference devices.
The CSV records these distinctions explicitly; unavailable parameter counts
are empty cells.

The complete model's improvements below are **percentage-point differences**,
calculated from the comparison values, rather than relative percentage gains:

| Comparison | mIoU gain | Mango F1 gain | Litchi F1 gain |
| --- | ---: | ---: | ---: |
| Full model versus without adapter | 0.43 | 0.63 | 0.12 |
| Full model versus without OCE | 0.88 | 0.69 | 0.81 |
| Full model versus U-Net | 2.67 | 2.50 | 1.73 |

The full model has the highest OA, kappa, and mIoU among the ten comparison
methods. Some pixel classifiers have higher mango F1; their litchi F1 is lower.
The larger mIoU decrease in the OCE ablation indicates a larger contribution
from orchard context under this fixed validation protocol.

### Row-normalized confusion proportions

Rows are reference classes and columns are predictions. Entries are rounded
proportions, not pixel counts; diagonal entries give class recall at the
reported precision.

| Reference | Predicted background | Predicted mango | Predicted litchi |
| --- | ---: | ---: | ---: |
| Background | 0.975 | 0.019 | 0.006 |
| Mango | 0.109 | 0.890 | 0.001 |
| Litchi | 0.052 | 0.000 | 0.948 |

## Model structure

1. **AEF feature adaptation:** a 64 -> 16 -> 64 low-rank residual adapter starts
   from an identity mapping, with bounded channel scaling of +/-10%.
2. **Spatial encoder:** a validity-mask channel accompanies the 64 AEF channels,
   giving 65 input channels. A 64-channel encoder and three depthwise-separable
   residual blocks preserve spatial resolution.
3. **Orchard Context Enhancement (OCE):** local 3 x 3 convolution and 7 x 7 / 15 x 15
   average-pooling branches capture 30-, 70-, and 150-m neighbourhood support on
   the 10-m grid. Pixelwise softmax weights combine the branches. Orchard and
   species paths receive different strengths of the resulting context residual.
4. **Hierarchical classification:** orchard probability `q` and conditional
   species probabilities `r_mango`, `r_litchi` yield `P(background)=1-q`,
   `P(mango)=q*r_mango`, and `P(litchi)=q*r_litchi`.
5. **Joint loss:** focal, Tversky, orchard auxiliary, species auxiliary, and
   hard-pixel species-margin losses supervise the three-class prediction.

### Implemented variants

| Code variant | AEF adapter | OCE | Hierarchical heads | Comparison mapping |
| --- | --- | --- | --- | --- |
| `C0_baseline` | No | No | No | Not included in the comparison |
| `C1_adapter` | Yes | No | No | Not included in the comparison |
| `C2_adapter_oce` | Yes | Yes | No | Not included in the comparison |
| `C3_hierarchical` | Yes | Yes | Yes | AEF-OCE-Net |
| `C3_minus_adapter` | No | Yes | Yes | AEF-OCE-Net without adapter |
| `C3_minus_oce` | Yes | No | Yes | AEF-OCE-Net without OCE |

This folder implements the AEF-OCE-Net variants. The comparison also includes
U-Net, DeepLabV3+, PSPNet, and pixel classifiers; their training implementations
are not included in this folder.

## Installation and data configuration

Use Python 3.10 or newer and install PyTorch for the available hardware.
Run from the **repository root**, one directory above this README:

```bash
python -m pip install -r requirements.txt
```

See the [dependency list](../requirements.txt). Dependencies include NumPy,
pandas, PyTorch, Zarr 2 (`zarr<3`), numcodecs, Matplotlib, and Rasterio.

Configure paths in [configuration.py](configuration.py). The default paths are
relative to the repository root:

- processed inputs: `data/patch_store_zarr_v1/`;
- fixed-split manifest: `config/fixed_split_manifest.csv`;
- run outputs: `outputs/fixed_split/<variant>/<run-name>/`.

`Config.source_npz` retains the original project's ancestor-directory layout.
Set it to the available NPZ source, or pass explicit `--source`, `--output`, and
`--manifest` paths to the preparation module. That module reads an existing
manifest and packs the original fixed NPZ split into Zarr with index files.
Training data, the manifest, and trained checkpoints must be supplied separately.
The code uses AEF embeddings; AEF foundation-model training and embedding
generation are outside this implementation.

## Running the code

Run modules from the repository root so their relative imports work. Python's
module launcher accepts the `AEF-OCE-Net` folder name:

```bash
python -m AEF-OCE-Net.prepare_fixed_split_zarr --help
python -m AEF-OCE-Net.train --help
python -m AEF-OCE-Net.train --variant C3_hierarchical --run-name experiment_01
```

Optional audit, smoke test, evaluation, and raster prediction:

```bash
python -m AEF-OCE-Net.audit_project
python -m AEF-OCE-Net.train --variant C3_hierarchical --run-name smoke_01 --device cpu --smoke-test
python -m AEF-OCE-Net.evaluate path/to/best.pt
python -m AEF-OCE-Net.predict_raster path/to/best.pt path/to/aef.tif path/to/predictions
```

Audits, smoke tests, and training require the external inputs and manifest.
Choose a new `--run-name` for each run; existing populated run directories are
protected against overwrite. Raster prediction requires a compatible checkpoint
and a 64-band annual AEF raster. A smoke test verifies a short training and
validation run; its score should not be used as a final accuracy estimate.

## Release contents

The release contains the AEF-OCE-Net source, its
[comparison and ablation results](model_accuracy.csv), and the
[2017-2025 annual maps and map accuracy results](../spatial_distribution_maps/README.md).
The model and annual-map evaluations retain their distinct reference data and
evaluation protocols.
