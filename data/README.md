# Study data

The original 510 recording files are stored as 123 split ZIP parts, each at
most 24,000,000 bytes. Filenames, directory structure and data contents are
preserved. File counts refer to recording files, not participant counts.

## Download and restore

1. Download the complete repository using **Code > Download ZIP** and extract
   the downloaded repository ZIP, or clone the repository. Download all data
   parts, `data_manifest.json`, and `restore_data.py` together.
2. Install Python 3.9 or newer if needed. No third-party packages are required.
3. Open a terminal at the extracted repository root and run:

   ```sh
   python data/restore_data.py --output .
   ```

   The script verifies every part, each reconstructed ZIP stream and every
   decompressed file, then restores:

   ```text
   data/50 participants database/
   data/24 participants skill learning data/
   data/additional posture data/
   ```

   Allow at least 7.3 GB of additional disk space for the restored data. The
   script reads the parts directly; no large intermediate ZIP files are created.
   Existing identical data files are skipped. Conflicting files stop restoration
   before any data files are written.

If you downloaded the parts separately into one folder, use:

```sh
python restore_data.py --output "PATH_TO_REPOSITORY"
```

To check downloads without extracting:

```sh
python data/restore_data.py --verify-only
```

These are split files, not independent ZIPs. Do not extract each part separately
or rename any part. A missing or altered part is reported by the restore script.
The original unsplit ZIPs are not needed.

## Integrity and organization

| Original cohort | Recording files | ZIP parts |
| --- | ---: | ---: |
| 50 participants database | 217 | 54 |
| 24 participants skill learning data | 180 | 49 |
| additional posture data | 113 | 20 |

`data_manifest.json` records SHA-256 checksums for all parts, ZIP streams and
original files. `SHA256SUMS.txt` also covers the companion files.

The `batch_*` directories in the local upload package only group files for
uploading. Upload their **contents** into this repository's `data/` directory,
one batch per upload. The restore script also accepts the local batched layout
with `--archives PATH_TO_UPLOAD_PACKAGE`.
