# SeismicWorld

**Generalizable End-to-End 4D Seismic Modeling via Variational Encoding and LLM-driven Latent Space Reasoning**

## Overview

SeismicWorld is a novel approach to 4D seismic evolution modeling that reframes the problem as latent token sequence reasoning instead of traditional voxel regression. By combining variational compression with finite scalar quantization (FSQ) and autoregressive LLM-based forecasting, this method achieves improved structural fidelity in time-lapse seismic predictions.

## Key Features

- **Latent Token Modeling**: Transforms 4D seismic data into discrete latent tokens for efficient reasoning
- **Variational Encoding**: Uses MagVit-V2 based architecture with finite scalar quantization (FSQ)
- **LLM-driven Forecasting**: Autoregressive language model predicts future seismic states
- **CHARM-4D Dataset**: Synthetic dataset for pretraining with diverse geological scenarios
- **Sleipner Benchmark**: Real-world validation on Sleipner CO2 storage field (2001-2010)

## Project Structure

```
SeismicWorld/
├── Code/
│   ├── MoE/                    # Main model implementations
│   │   ├── magvit_v2.py       # Variational autoencoder
│   │   ├── magvit_v2_ldm.py   # Latent diffusion model
│   │   ├── modeling_internlm.py # LLM backbone
│   │   ├── functions.py        # Utility functions
│   │   └── models/             # Jupyter notebooks for training/inference
│   │       ├── SeismicWorld.ipynb
│   │       ├── GeoWorld.ipynb
│   │       └── fine_tune.ipynb
│   ├── pted_lm/               # Pretrained language model components
│   ├── syn_creation/          # Synthetic data generation
│   └── scripts/               # Training and evaluation scripts
├── Paper/                      # Research paper and figures
├── Paper_geophysics/           # Journal submission materials
└── Agents.md                   # Author-agent collaboration guide

```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/SeismicWorld.git
cd SeismicWorld

# Install dependencies (create requirements.txt as needed)
pip install torch torchvision numpy scipy matplotlib jupyter
pip install segysak ObsPy  # For seismic data processing
```

## Usage

### Training

Training notebooks are located in `Code/MoE/models/`:

1. **Pretraining on CHARM-4D**: Use `SeismicWorld.ipynb`
2. **Fine-tuning on Sleipner**: Use `fine_tune.ipynb`

### Inference

For inference and visualization:

```python
# Example coming soon - see notebooks for detailed examples
```

## Dataset

### CHARM-4D (Synthetic)
- **Purpose**: Pretraining dataset with diverse geological scenarios
- **Size**: 2,000 sequences with timesteps T ∈ [6, 30]
- **Scenarios**: CO2 injection, waterflooding, gas migration
- **Availability**: [Dataset link to be added]

### Sleipner Field Data
- **Purpose**: Real-world benchmark evaluation
- **Period**: 2001-2010 time-lapse seismic surveys
- **Usage**: Low-shot fine-tuning and blind evaluation
- **Preprocessing**: Contact for access terms

## Model Checkpoints

Due to GitHub file size limitations, model checkpoints are not included in this repository. You can:

1. Train your own models using the provided notebooks
2. Download pretrained checkpoints from: [Link to be added]

Available checkpoints:
- `seismic_world_best_finetuned.pth` (~2.5GB)
- `vqvae_trained_best.pth` (~1.2GB)

## Evaluation Metrics

The model is evaluated using:
- **Structural Fidelity**: SSIM, RASC (Reflector Amplitude and Structural Consistency)
- **Amplitude Accuracy**: MAE, PSNR
- **Cross-correlation**: Temporal coherence analysis

## Paper

This work is under submission. For detailed methodology and results, please refer to the paper in the `Paper/` directory.

**Citation** (to be updated upon publication):
```bibtex
@article{seismicworld2025,
  title={SeismicWorld: Generalizable End-to-End 4D Seismic Modeling via Variational Encoding and LLM-driven Latent Space Reasoning},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```

## Reproducibility

- Training/inference code: `Code/MoE/`
- Exact evaluation protocols: See paper appendix
- Random seeds and hyperparameters: Documented in notebooks

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

[License to be added - e.g., MIT, Apache 2.0, or GPL]

## Contact

For questions or collaboration inquiries, please open an issue or contact [your email].

## Acknowledgments

- Based on MagVit-V2 architecture
- Sleipner dataset provided by [data provider]
- Supported by [funding sources if applicable]
