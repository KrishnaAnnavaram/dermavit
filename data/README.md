# Data

The repository does not contain images or metadata. Git ignores everything in `/data/` except this README.

## HAM10000 (training and validation)

| Item | Value |
|---|---|
| Source | "The HAM10000 dataset", Harvard Dataverse, doi:10.7910/DVN/DBW86T, `https://doi.org/10.7910/DVN/DBW86T` |
| License | CC BY-NC 4.0 (non-commercial). Cite Tschandl et al., Sci. Data 5, 180161 (2018) |
| Size | 10,015 images, about 3 GB |
| Metadata columns | `lesion_id`, `image_id`, `dx`, `dx_type`, `age`, `sex`, `localization` |
| Classes (`dx`) | `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc` |

## ISIC 2018 Task 3 test set (external test)

| Item | Value |
|---|---|
| Source | ISIC 2018 Challenge, `https://challenge.isic-archive.com/data/#2018` |
| License | CC BY-NC 4.0, see the ISIC terms of use |
| Files | `ISIC2018_Task3_Test_Images.zip` (1,512 images), `ISIC2018_Task3_Test_GroundTruth.csv` (columns `image, MEL, NV, BCC, AKIEC, BKL, DF, VASC`) |

## Expected layout

```
data/
├── HAM10000_metadata.csv
├── images/                              # all HAM10000 JPEGs (or HAM10000_images_part_1/ and _part_2/)
├── ISIC2018_Task3_Test_Images/          # test JPEGs
└── ISIC2018_Task3_Test_GroundTruth.csv
```

1. Download the Dataverse files. Unzip the two image parts into `data/images/`.
2. Rename the metadata file to `HAM10000_metadata.csv` if it has no extension.
3. Download the Task 3 test images and ground truth from the ISIC archive. Unzip the images into `data/ISIC2018_Task3_Test_Images/`.
4. Run `dermavit validate data/HAM10000_metadata.csv`.

`lesion_id` is necessary: the split groups images by lesion.

## Synthetic data (no download)

```bash
dermavit synth --out data/synthetic      # same layout, small PNG images, SYNTHETIC lesions
dermavit demo                            # in memory, nothing is written
```

The synthetic generator never reads real images. The tests and the offline demo use only synthetic data.
