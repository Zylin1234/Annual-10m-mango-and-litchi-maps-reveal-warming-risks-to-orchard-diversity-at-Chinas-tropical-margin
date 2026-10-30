# AEF-OCE-Net

PyTorch source code for the refined AEF-OCE-Net model for pixelwise background,
mango, and lychee classification from precomputed AlphaEarth Foundations (AEF)
embeddings. The default complete model is `C3_hierarchical`.

## Model

- Input: 64 AEF embedding channels plus one validity-mask channel (65 channels in total).
- Identity-initialized, low-rank AEF residual adapter (rank 16).
- A resolution-preserving encoder with 64 feature channels and three depthwise residual blocks.
- Dual-path Orchard Context Enhancement (OCE): local 3x3 convolution and 7x7/15x15
  average-pooling branches, with pixelwise branch weights and separate orchard/species context strengths.
- Hierarchical orchard and conditional-species heads: `P(background)=1-q`,
  `P(mango)=q*r_mango`, and `P(lychee)=q*r_lychee`.
- Joint focal, Tversky, orchard auxiliary, species auxiliary, and hard-pixel species-margin losses.

The six implemented variants are:

| Variant | AEF adapter | OCE | Hierarchical heads |
| --- | --- | --- | --- |
| `C0_baseline` | No | No | No |
| `C1_adapter` | Yes | No | No |
| `C2_adapter_oce` | Yes | Yes | No |
| `C3_hierarchical` | Yes | Yes | Yes |
| `C3_minus_adapter` | No | Yes | Yes |
| `C3_minus_oce` | Yes | No | Yes |

## Scope and data protocol

This source snapshot uses the original fixed `fold0` allocation: 20 training plots
(5,752 patches) and six validation plots (409 patches). It does not implement
three-fold cross-validation, despite the name of the original local folder.
The split includes counties represented in both training and validation and
must not be described as county-independent validation.

The repository contains the unchanged Python sources and their dependency list.
Training data, model weights, and the fixed-split manifest are not included.
AEF embeddings must be obtained separately; this code does not train the AEF
foundation model or generate its satellite embeddings. This snapshot alone does
not provide complete reproduction of all manuscript experiments and results.

## Annual spatial distribution maps

Hainan mango and lychee prediction maps for 2017-2025 are available as GeoTIFFs in
[`spatial_distribution_maps/`](spatial_distribution_maps/). All nine files follow
the naming convention `Hainan_<year>_mango_lychee_AEF_OCE_pred.tif`.
The maps retain their original pixel values and geospatial metadata; only the
published filenames have been standardized. See the folder README for file
details and `checksums.sha256` for integrity verification.

## Setup

Use Python 3.10 or newer and a PyTorch installation appropriate for your hardware.
The dependency list is copied from the original project and does not pin every
package version.

```bash
python -m pip install -r requirements.txt
```

Dependencies: NumPy, pandas, PyTorch, Zarr 2 (`zarr<3`), numcodecs, Matplotlib, and Rasterio.

Before training, provide the processed Zarr input and the fixed-split manifest
at `config/fixed_split_manifest.csv`, or configure their locations in
`src/configuration.py`. Review `Config.source_npz`, `Config.data_path`,
`Config.manifest_path`, and `Config.output_root`: the default source path assumes
the original local directory layout. The manifest and data must follow the
schema expected by `src/data.py`. The preparation module reads an existing
fixed-split manifest and packs the original NPZ split into Zarr with index files.

## Run

Run modules from the repository root so package-relative imports work:

```bash
python -m src.prepare_fixed_split_zarr --help
python -m src.train --help
python -m src.train --variant C3_hierarchical --run-name experiment_01
```

Optional checks, evaluation, and raster prediction:

```bash
python -m src.audit_project
python -m src.train --variant C3_hierarchical --run-name smoke_01 --device cpu --smoke-test
python -m src.evaluate path/to/best.pt
python -m src.predict_raster path/to/best.pt path/to/aef.tif path/to/predictions
```

The audit, smoke test, and training require the external data and manifest.
Use a new `--run-name` for each training run; existing output directories are
protected against overwrite. Outputs are written under
`outputs/fixed_split/<variant>/<run-name>/` by default. Prediction requires a
compatible checkpoint and a 64-band AEF raster.
