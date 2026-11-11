# Public data sources and local file contract

The dataset arrays are not committed to this repository. They remain ignored
by `.gitignore`; users must obtain the public benchmark data and prepare the
local replay inputs before running the frozen evaluation.

## Public acquisition links

All five benchmarks originate from the California Department of
Transportation Performance Measurement System (Caltrans PeMS). The
[Caltrans PeMS archive](https://pems.dot.ca.gov/) is the original data service
and requires a free account. The benchmark subsets commonly used in traffic
forecasting papers are easier to obtain from the benchmark authors or the
public mirrors below.

| Public name | Local ID | Required `data.npz` shape | Benchmark-author source | Public download |
| --- | --- | ---: | --- | --- |
| PEMS03 | `PEMS03` | `26208 x 358 x 3` | [STSGCN repository](https://github.com/Davidham3/STSGCN) | [Author-maintained Hugging Face dataset](https://huggingface.co/datasets/Davidham3/pems-stsgcn/tree/main/PEMS03); [Zenodo NPZ mirror](https://doi.org/10.5281/zenodo.7816008) |
| PEMS04 | `PEMS04` | `16992 x 307 x 3` | [ASTGCN repository](https://github.com/Davidham3/ASTGCN-2019-mxnet) | [Author-maintained Hugging Face dataset](https://huggingface.co/datasets/Davidham3/pems-stsgcn/tree/main/PEMS04); [Zenodo NPZ mirror](https://doi.org/10.5281/zenodo.7816008) |
| PEMS07 | `PEMS07` | `28224 x 883 x 3` | [STSGCN repository](https://github.com/Davidham3/STSGCN) | [Author-maintained Hugging Face dataset](https://huggingface.co/datasets/Davidham3/pems-stsgcn/tree/main/PEMS07); [Zenodo NPZ mirror](https://doi.org/10.5281/zenodo.7816008) |
| PEMS08 | `PEMS08` | `17856 x 170 x 3` | [ASTGCN repository](https://github.com/Davidham3/ASTGCN-2019-mxnet) | [Author-maintained Hugging Face dataset](https://huggingface.co/datasets/Davidham3/pems-stsgcn/tree/main/PEMS08); [Zenodo NPZ mirror](https://doi.org/10.5281/zenodo.7816008) |
| PEMS-BAY | `PEMSBAY` | `52116 x 325 x 3` | [DCRNN repository and preparation instructions](https://github.com/liyaguang/DCRNN#data-preparation) | [DCRNN authors' public Google Drive folder](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) (`pems-bay.h5`) |

The first four Hugging Face downloads are Parquet releases maintained by the
STSGCN benchmark author. The Zenodo record is a convenient NPZ mirror and
publishes MD5 values for its files. It is a mirror, not the original Caltrans
service. PEMS-BAY should be taken from the DCRNN authors' link above.

Do not substitute a PEMS-BAY package with 52,128 timestamps: the frozen replay
uses 52,116 timestamps. A different length changes the split boundaries and
is not compatible with the retained index/scaler file.

## Required local layout

The public downloads are source data, not drop-in replacements for all 15
prepared replay inputs. The frozen replay requires this exact local layout:

```text
datasets/
  PEMS03/
    data.npz
    PEMS03_time.npz
    PEMS03_12to12_index_scaler.npz
  PEMS04/
    data.npz
    PEMS04_time.npz
    PEMS04_12to12_index_scaler.npz
  PEMS07/
    data.npz
    PEMS07_time.npz
    PEMS07_12to12_index_scaler.npz
  PEMS08/
    data.npz
    PEMS08_time.npz
    PEMS08_12to12_index_scaler.npz
  PEMSBAY/
    data.npz
    PEMSBAY_time.npz
    PEMSBAY_12to12_index_scaler_712.npz
```

`data.npz` stores the benchmark signal. The `*_time.npz` and
`*_index_scaler*.npz` files additionally encode the timestamp features,
frozen split indices, and training-set scaler used by this package. A newly
downloaded source file must not be assumed equivalent merely because its name
or dimensions match.

If the exact prepared files already exist in a trusted workspace, copy only
those files with the repository utility:

```bash
python -B tools/prepare_local_inputs.py --source <trusted-workspace>
python -B tools/prepare_local_inputs.py --source <trusted-workspace> --apply
python -B tools/preflight.py
```

The first command is a preview. `tools/preflight.py` checks the required files
and frozen asset hashes; it does not certify that an independently converted
dataset has the same numerical contents as the paper preparation.

## Redistribution and attribution

The links above do not transfer ownership of the datasets to this repository.
Before redistributing downloaded data, consult the source provider's terms and
cite the corresponding benchmark paper or repository. This package therefore
keeps prepared dataset arrays out of Git while retaining source attribution
and acquisition instructions.
