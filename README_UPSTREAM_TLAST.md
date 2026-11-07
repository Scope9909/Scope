# TLAST
This is a Pytorch implementation of TLAST: A Time-Lag Aware Spatial-Temporal Transformer for Traffic Flow Forecasting (TITS 2025). [[Paper](https://ieeexplore.ieee.org/document/11077815)]

## Data preparation
Following the detail descriptions [here](datasets/readme.md).


## Requirements
Python 3.9.12, torch 1.13.1+cu117, numpy 1.26.4, einops 0.8.0

## Usage
Four road network datasets:
```shell
for config in 'PEMS04_12_12' 'PEMS07_12_12' 'PEMS08_12_12' 'CA_12_12'; do
python run.py --config_file $config.json --modelid TLAST --device cuda:0
done
```

Three city grid datasets:
```shell
for config in 'CHIBike_6_1' 'NYCTaxi_6_1' 'T-Drive_6_1'; do
python run.py --config_file $config.json --modelid TLAST --device cuda:0
done
```

The hyper-parameter settings for all datasets are given in `.configurations/` folder.

Results will be saved in the `EXP_results` folder.


## Acknowledgement
Thanks to the following inspiring research and their valuable codes: [PDFormer](https://github.com/BUAABIGSCity/PDFormer), [LargeST](https://github.com/liuxu77/LargeST), [DIFFormer](https://github.com/qitianwu/DIFFormer), [Informer](https://github.com/zhouhaoyi/Informer2020)

## Cite
If you find this project helpful, please cite us:
```bibtex
@article{TLAST,
  author = {Zheng, Qi and Shao, Minhua and Zhang, Yaying},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  title = {TLAST: A Time-Lag Aware Spatial-Temporal Transformer for Traffic Flow Forecasting},
  year = {2025},
  volume = {26},
  number = {9},
  pages = {13144-13158},
  doi = {10.1109/TITS.2025.3583391},
}
```
