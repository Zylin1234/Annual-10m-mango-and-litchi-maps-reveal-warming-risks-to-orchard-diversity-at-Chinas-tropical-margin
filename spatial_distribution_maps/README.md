# Hainan mango and lychee spatial distribution maps (2017-2025)

This folder contains nine annual GeoTIFF prediction maps supplied for the
AEF-OCE-Net repository, one for each year from 2017 through 2025.

## Filenames

All maps use `Hainan_<year>_mango_lychee_AEF_OCE_pred.tif`.
The 2017 and 2019 source files were named `Hainan_2017.tif` and
`Hainan_2019.tif`; their published filenames have been standardized.
Each published TIFF is byte-for-byte identical to its source. Pixel values,
compression, georeferencing, and embedded metadata have been preserved.

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

## Integrity verification

`checksums.sha256` records the SHA-256 hash of each TIFF under its published
filename. All source/copy hashes were checked before upload.

From this folder, verify downloaded files with:

```bash
sha256sum -c checksums.sha256
```

On Windows PowerShell, `Get-FileHash -Algorithm SHA256 <filename>` can be used
to compare an individual file with the checksum list.
