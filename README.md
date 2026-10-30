# Annual 10-m mango and litchi maps reveal warming risks to orchard diversity at China's tropical margin

Annual mango and litchi mapping in Hainan, China, using AlphaEarth Foundations
(AEF) embeddings and AEF-OCE-Net. This repository provides the model source,
nine annual GeoTIFF maps for **2017–2025**, and the final accuracy results for
the study.

The maps support analysis of orchard distributions at China's tropical margin.
AEF-OCE-Net combines low-rank feature adaptation, Orchard Context Enhancement
(OCE), and hierarchical orchard/species classification. The nominal mapping
resolution is **10 m**.

## Repository guide

| Resource | Contents |
| --- | --- |
| [AEF-OCE-Net](AEF-OCE-Net/README.md) | Model source, architecture, installation and running instructions, internal validation results, comparisons, and ablations |
| [Spatial distribution maps](spatial_distribution_maps/README.md) | The nine annual maps, year-specific accuracy, raster metadata, class codes, and download links |
| [Model accuracy CSV](AEF-OCE-Net/model_accuracy.csv) | Model comparison and ablation metrics in a machine-readable form |
| [Annual map accuracy CSV](spatial_distribution_maps/annual_map_accuracy.csv) | All six accuracy metrics for each year |
| [Map checksums](spatial_distribution_maps/checksums.sha256) | SHA-256 checksums for the nine GeoTIFFs |
| [Dependencies](requirements.txt) | Python packages used by the implementation |

```text
.
├── README.md
├── requirements.txt
├── AEF-OCE-Net/
│   ├── README.md
│   ├── model_accuracy.csv
│   └── *.py
└── spatial_distribution_maps/
    ├── README.md
    ├── annual_map_accuracy.csv
    ├── checksums.sha256
    └── Hainan_<year>_mango_lychee_AEF_OCE_pred.tif
```

“Litchi” in the documentation and “lychee” in existing filenames and code identifiers
refer to the same crop. Those filenames and identifiers are retained for
compatibility.

## Accuracy results

The model and annual maps use distinct evaluation protocols, documented in their
corresponding folders:

| Evaluation | Reference data | Key result | Details |
| --- | --- | --- | --- |
| AEF-OCE-Net internal model validation (2025) | Six validation plots; 409 image tiles; model trained on 20 separate plots and 5,752 tiles | OA **96.26%**, kappa **0.8850**, mIoU **0.8750**, mango F1 **0.8828**, litchi F1 **0.9362** | [Model results](AEF-OCE-Net/README.md#internal-validation-results) |
| Annual map assessment (2017–2025) | 1,730 year-labelled reference points per year; 15,570 point-year records | Mean annual OA **94.58%**; annual OA **90.00–98.55%**; 2025 map OA **98.55%** | [Annual results](spatial_distribution_maps/README.md) |

The internal 2025 model result and the 2025 annual map result use different
reference data and evaluation units. The complete model source supplied here
uses a fixed training/validation allocation.

## Use the maps

Open [spatial_distribution_maps](spatial_distribution_maps/README.md) to select a
year. Every map follows `Hainan_<year>_mango_lychee_AEF_OCE_pred.tif` and contains
one classified raster band. Class codes are **0 = background**, **1 = mango**,
and **2 = litchi**. The folder README explains the NoData metadata differences
between years and how to verify downloaded files.

The GeoTIFFs retain the source pixel values and geospatial metadata. Their
checksums allow byte-level verification.

## Use the model

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m AEF-OCE-Net.train --help
```

Training and evaluation require external data and the fixed-split manifest.
Configure input paths in [AEF-OCE-Net/configuration.py](AEF-OCE-Net/configuration.py)
before running experiments. See the [model README](AEF-OCE-Net/README.md) for
preparation, training, ablation, evaluation, and raster-inference commands.

## Release contents

This final release contains the AEF-OCE-Net source, the nine
annual map products, and the documented accuracy tables. Model checkpoints,
training labels, the fixed-split manifest, annual reference-point records, and
the complete climate-analysis workflow are not included. AEF embeddings must be
obtained separately from their provider. See the model and map READMEs for
evaluation methods, input requirements, and usage instructions.
