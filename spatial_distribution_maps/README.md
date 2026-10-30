# Hainan mango and litchi spatial distribution maps (2017-2025)

This folder contains nine annual GeoTIFF prediction maps for the study
*Annual 10-m mango and litchi maps reveal warming risks to orchard diversity at
China's tropical margin*, one for each year from 2017 through 2025.
"Litchi" in the documentation and "lychee" in filenames refer to the same crop.

## Raster metadata

All nine files have these embedded properties:

| Property | Value |
| --- | --- |
| Coordinate reference system | EPSG:4326 (WGS 84) |
| Width x height | 29,953 x 22,507 pixels |
| Bands | 1 |
| Data type | uint8 |

NoData metadata differs by year and is retained as supplied:

| Years | Embedded NoData value | Values observed in a 512 x 512 overview sample |
| --- | --- | --- |
| 2017-2022 | Not declared | 0, 1, 2 |
| 2023-2025 | 255 | 0, 1, 2, 255 |

The sample values are a spot check, rather than an exhaustive unique-value scan.
The source model uses class codes 0 = background, 1 = mango, and 2 = lychee.
Open the TIFFs in GIS software with a categorical renderer. Respect each file's
own NoData metadata when computing areas or combining years.

## Annual map accuracy

The final annual maps achieve a **mean annual overall accuracy (OA)
of 94.58%**, with annual OA ranging from **0.9000 to 0.9855**. The same selected
model, trained with 2025 AEF data and plot labels, was applied to the annual AEF
layers for 2017-2025.

Each annual map was evaluated against **1,730 year-specific reference points**,
for **15,570 sample-point records** across nine years. Labels were established
using field surveys and multispectral and high-resolution imagery. The 2025
reference sample contained **405 litchi, 455 mango, and 870 background points**;
these class counts describe 2025 only.

These annual product metrics use independent, year-specific reference points.
They are separate from the model's
[internal 2025 plot-validation and ablation results](../AEF-OCE-Net/README.md#internal-validation-results);
the two evaluation sets and their accuracy values are not interchangeable.
The table gives accuracy values to four decimal places on a 0-1 scale.

| Year / GeoTIFF | OA | Kappa | Macro-F1 | Weighted-F1 | Mango F1 | Litchi F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [2017](Hainan_2017_mango_lychee_AEF_OCE_pred.tif) | 0.9000 | 0.8194 | 0.8798 | 0.9004 | 0.9125 | 0.8056 |
| [2018](Hainan_2018_mango_lychee_AEF_OCE_pred.tif) | 0.9370 | 0.8921 | 0.9326 | 0.9372 | 0.9362 | 0.9170 |
| [2019](Hainan_2019_mango_lychee_AEF_OCE_pred.tif) | 0.9150 | 0.8591 | 0.9090 | 0.9163 | 0.9333 | 0.8698 |
| [2020](Hainan_2020_mango_lychee_AEF_OCE_pred.tif) | 0.9168 | 0.8626 | 0.9129 | 0.9172 | 0.9176 | 0.8967 |
| [2021](Hainan_2021_mango_lychee_AEF_OCE_pred.tif) | 0.9601 | 0.9360 | 0.9593 | 0.9602 | 0.9672 | 0.9499 |
| [2022](Hainan_2022_mango_lychee_AEF_OCE_pred.tif) | 0.9746 | 0.9590 | 0.9741 | 0.9746 | 0.9749 | 0.9719 |
| [2023](Hainan_2023_mango_lychee_AEF_OCE_pred.tif) | 0.9555 | 0.9274 | 0.9548 | 0.9556 | 0.9484 | 0.9580 |
| [2024](Hainan_2024_mango_lychee_AEF_OCE_pred.tif) | 0.9676 | 0.9479 | 0.9670 | 0.9676 | 0.9719 | 0.9605 |
| [2025](Hainan_2025_mango_lychee_AEF_OCE_pred.tif) | 0.9855 | 0.9768 | 0.9856 | 0.9856 | 0.9825 | 0.9888 |

A machine-readable version, including the validation-point count for each year,
is available in [`annual_map_accuracy.csv`](annual_map_accuracy.csv).

### Metric definitions

- **OA:** the proportion of reference points classified correctly.
- **Kappa:** overall agreement corrected for the agreement expected by chance.
- **Macro-F1:** the unweighted mean of background, mango, and litchi F1 scores.
- **Weighted-F1:** the mean of the three class F1 scores weighted by each class's
  frequency in the reference sample for that year.
- **Mango F1:** the harmonic mean of mango precision (user's accuracy) and recall
  (producer's accuracy).
- **Litchi F1:** the harmonic mean of litchi precision (user's accuracy) and recall
  (producer's accuracy).

Confusion-matrix rows represent reference classes and columns represent mapped
classes. Precision uses the mapped-class total as its denominator; recall uses
the reference-class total. No year-specific precision or recall values are
inferred from the F1 scores here.

## Integrity verification

`checksums.sha256` records the SHA-256 hash of each TIFF under its published
filename. All source/copy hashes were checked before upload.

From this folder, verify downloaded files with:

```bash
sha256sum -c checksums.sha256
```

On Windows PowerShell, `Get-FileHash -Algorithm SHA256 <filename>` can be used
to compare an individual file with the checksum list.
