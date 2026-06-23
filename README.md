# HybridRAFT: Hybrid Retrieval-Augmented Forecasting Network for Traffic Flow Prediction

[![DOI](https://img.shields.io/badge/DOI-zenodo.20806264-blue.svg)](https://doi.org/10.5281/zenodo.20806264) 

**HybridRAFT** is a deep learning model for **long-term multivariate time series forecasting**. It builds upon the RAFT (Retrieval-Augmented Forecasting Transformer) backbone and introduces three auxiliary modules to enhance predictive signals through a *main-auxiliary augmentation architecture*.

## Architecture

HybridRAFT consists of four components:

| Module | Description |
|--------|-------------|
| **Event Contextualizer** | Encodes timestamp features (month, day, weekday, hour) using embeddings and a Gated Residual Network (GRN) to capture temporal dependencies. |
| **Periodic Profiler** | Extracts Top-K periodic patterns via FFT and applies multi-scale Inception convolutions to model periodic structure. |
| **Dynamic Graph Encoder** | Learns a dynamic spatial adjacency matrix from static node embeddings and time-varying correlations, then applies graph convolution. |
| **RAFT Backbone** | Dual-path prediction: (1) direct linear projection, and (2) retrieval-based prediction using multi-granularity periodic similarity search over historical patterns. |

The four modules work together:
1. **Signal augmentation**: Raw input is enhanced by Event Contextualizer and Periodic Profiler outputs.
2. **Spatial modeling**: The Dynamic Graph Encoder captures evolving inter-variable relationships.
3. **Dual-path forecasting**: RAFT produces both a direct forecast and a similarity-retrieved forecast.
4. **Fusion**: The two predictions are combined via a learned linear layer, with graph adjacency injected into the retrieval path.

## Results on PEMS04

The PEMS04 dataset contains traffic flow measurements from 307 sensors in the San Francisco Bay Area. Input length is fixed at 96 time steps.

| Pred Len | MSE | MAE |
|----------|-----|-----|
| 12 | 0.0785 | 0.1810 |
| 24 | 0.1365 | 0.2557 |
| 48 | 0.1489 | 0.2874 |
| 96 | 0.1121 | 0.2257 |

## Project Structure

```
HybridRAFT/
├── run.py                          # Main entry point
├── models/
│   └── HybridRAFT.py               # Core model definition
├── layers/
│   └── Retrieval.py                # RAFT retrieval tool
├── data_provider/
│   ├── data_factory.py             # DataLoader factory
│   └── data_loader.py              # Dataset classes (ETT + custom)
├── exp/
│   ├── exp_basic.py                # Experiment base class
│   └── exp_long_term_forecasting.py # Train/val/test pipeline
├── utils/
│   ├── augmentation.py             # Time series augmentation
│   ├── dtw.py                      # DTW distance
│   ├── dtw_metric.py               # Accelerated DTW (scipy)
│   ├── metrics.py                  # MSE, MAE, RMSE, MAPE, MSPE
│   ├── timefeatures.py             # Time feature encoding (GluonTS)
│   ├── tools.py                    # LR scheduling, early stopping, plotting
│   └── print_args.py               # Argument printer
└── train_pems04.ps1                # Example training script (PEMS04)
```

## Dependencies

- Python >= 3.9
- PyTorch
- NumPy
- Pandas
- scikit-learn
- SciPy
- Matplotlib
- tqdm
- sktime

## Quick Start

### 1. Prepare data

Place your dataset CSV in a `data/` directory. For PEMS04:

```
../data/PEMS04/PEMS04.csv
```

The CSV should have the first column as the date/time index and subsequent columns as variables.

### 2. Train

```bash
python run.py \
    --is_training 1 \
    --root_path ./data/PEMS04/ \
    --data_path PEMS04.csv \
    --model HybridRAFT \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 12 \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307 \
    --d_model 512 \
    --d_ff 2048 \
    --n_heads 8 \
    --e_layers 2 \
    --d_layers 1 \
    --train_epochs 30 \
    --batch_size 64 \
    --learning_rate 0.0001 \
    --patience 5 \
    --loss MAE \
    --lradj cosine
```

### 3. Test

```bash
python run.py \
    --is_training 0 \
    --root_path ./data/PEMS04/ \
    --data_path PEMS04.csv \
    --model HybridRAFT \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 12 \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307 \
    --d_model 512 \
    --d_ff 2048 \
    --n_heads 8 \
    --e_layers 2 \
    --d_layers 1 \
    --batch_size 64
```

## Supported Datasets

- **ETT** (Electricity Transformer Temperature): ETTh1, ETTh2, ETTm1, ETTm2
- **PEMS** (traffic flow): PEMS04 (307 sensors)
- **Custom datasets**: CSV files with date index and variable columns

## Citation

This paper has been accepted by **IEEE SMC 2026** ([Link](https://www.ieeesmc2026.org/)). If you use HybridRAFT in your research, please cite:

```bibtex
@inproceedings{qiu2026hybridraft,
  title = {Hybrid Retrieval-Augmented Forecasting Network for Traffic Flow Prediction},
  author={Qiu, Jingxiang and Qu, Guanheng and Li, Jianrui and Liu, Jiangming},
  journal={},
  year={2026},
  address={Bellevue, WA, USA},
  month={October},
  url={},
  doi = {10.5281/zenodo.20806264},
  note={Accepted for publication},
}
```

## License

This project is open-source and available under the MIT License.
