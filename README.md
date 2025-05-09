[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15375049.svg)](https://doi.org/10.5281/zenodo.15375049)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-green.svg)](https://opensource.org/licenses/GPL-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-red.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-Science%20Direct-orange.svg)](https://)

# AutoFHR: A Neural Temporal Model for Fetal Cardiac Activity Analysis

AutoFHR is a GenAI-based solution for automatic heart rate localization from Doppler ultrasound (DUS) signals. It leverages an autoregressive architecture with dilated causal convolutions and attention mechanisms, as well as an innovative learning objective to specifically analyze fetal heart rate dynamics.

## Key Features

- **Architecture**: Utilizes causal dilated convolutions and residual blocks to capture long-range temporal dependencies
- **Attention Mechanism**: Employs multi-head attention to focus on relevant parts of the signal
- **Periodicity Loss**: Custom loss function that minimizes generation error while uniquely incorporating a spectral fidelity term to retain the natural rhythm of cardiac activity

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/cliffordlab/AutoFHR-Leveraging-Neural-Sequential-Modeling-for-Ultrasound-Analysis.git
cd AutoFHR
pip install -r requirements.txt
```

## Data

The model is designed to work with 1-dimensional DUS signals. The input data should be formatted as file with the following structure:

- `label`: Ground truth annotations
- `tensor_all`: Preprocessed tensors (scalograms)
- `beatset_list`: List of beat locations
- `FHR_list`: List of FHR values
- `DUS_list`: List of Doppler ultrasound signals

## Usage

### Training

#### Local Training

To train the model on your local machine:

```bash
python src/train.py --data_path /path/to/your/data.npz --output_dir ./output
```

Additional training options:
```
--filters: Number of filters in the model (default: 40)
--kernel_size: Kernel size in the model (default: 20)
--dilation_layers: Number of dilation layers in the model (default: 8)
--num_heads: Number of attention heads (default: 8)
--batch_size: Batch size for training (default: 64)
--epochs: Number of epochs to train (default: 1000)
--learning_rate: Learning rate for optimizer (default: 0.005)
--patience: Patience for early stopping (default: 100)
--use_generator: Use data generator for training
--random_seed: Random seed for reproducibility (default: 2025)
--gpu: GPU device to use (use -1 for CPU) (default: 0)

# Visualization options
--num_vis_samples: Number of samples to visualize (default: 4)
--vis_attention: Visualize attention weights
```

#### Cluster Training with Slurm

If you have access to a computing cluster that uses Slurm, you can use the provided Slurm script:

1. First, ensure that the logs directory exists:
```bash
mkdir -p logs
```

2. Modify the Slurm script (`slurm/run.sh`) to specify the correct paths and parameters for your environment

3. Submit the job to the Slurm scheduler:
```bash
sbatch slurm/run.sh
```

4. Monitor your job:
```bash
squeue -u <your_username>
```

5. Check logs:
```bash
cat logs/AutoFHR_job-<job_id>.out
```

### Prediction

To make predictions with a trained model:

```bash
python src/predict.py --model_path /path/to/model.h5 --input_file /path/to/signal.wav
```

Additional options:
```
--output_dir: Directory to save outputs (default: ./predictions)
--sampling_rate: Sampling rate of the input signal in Hz (default: 4000)
--time_bins: Number of time bins for the scalogram (default: 1000)
--freq_bins: Number of frequency bins for the scalogram (default: 40)
--gpu: GPU device to use (use -1 for CPU) (default: 0)
```

## Project Structure

```
AutoFHR/
├── data/                      # Sample data
├── autocorrelation/           # Autocorrelation function model
│   └── main.py                # Main implementation
├── slurm/                     # Slurm scripts for cluster training
│   └── run.sh                 # Main Slurm job script
├── src/                       # Source code
│   ├── data/                  # Data handling
│   │   └── data_loader.py     # Data loading utilities
│   ├── models/                # Model definitions
│   │   ├── model.py           # Main model architecture
│   │   ├── losses.py          # Custom loss functions
│   │   └── trainer.py         # Model training utilities
│   ├── utils/                 # Utility functions
│   │   ├── signal_processing.py # Signal processing utilities
│   │   ├── evaluation.py      # Evaluation metrics and plotting
│   │   └── visualization.py   # Advanced visualization utilities
│   ├── train.py               # Training script
│   └── predict.py             # Prediction script
├── tests/                     # Unit tests
├── output/                    # Model outputs (created when training)
│   └── visualizations/        # Generated visualizations
├── logs/                      # Slurm log files
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## License

This project is licensed under the GPL-3.0 License - see the LICENSE file for details.

## Citation

If you find this work useful, please cite:

```bibtex
@article{,
  author = {},
  title = {},
  year = {},
  url = {}
}
```

```bibtex
@software{rafiei_2025_15375049,
  title        = {AutoFHR v1.0.0},
  author       = {Rafiei, Alireza and Katebi, Nasim and Clifford, Gari D.},
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.15375049},
  url          = {https://doi.org/10.5281/zenodo.15375049},
}
```

To cite the autocorrelation function method, use:

```bibtex
@article{valderrama2019open,
  title         = {An open source autocorrelation-based method for fetal heart rate estimation from one-dimensional Doppler ultrasound},
  author        = {Valderrama, Camilo E and Stroux, Lisa and Katebi, Nasim and Paljug, Elianna and Hall-Clifford, Rachel and Rohloff, Peter and Marzbanrad, Faezeh and Clifford, Gari D},
  journal       = {Physiological measurement},
  volume        = {40},
  number        = {2},
  pages         = {025005},
  year          = {2019},
  publisher     = {IOP Publishing},
  doi           = {10.1088/1361-6579/ab033d}
}
```
