# PIAS-SSM

**Physics-Informed Adaptive Spectral with State Space Model** -- a few-shot rolling-bearing
fault diagnosis framework for strong-noise conditions.

## Architecture

```
Input (2-channel 1-D signal: vibration + instantaneous speed)
        |
  +-----v-----------+
  | Slicing operator |  <- keeps the vibration channel only
  +-----+-----------+
        |
  +-----v-------------------+
  | Learnable Morlet        |  <- 64 complex Morlet kernels, trainable centres,
  | frontend                |     log-domain scales, energy-normalised
  +-----+-------------------+
        |
  +-----v----+
  | AvgPool  |  <- pre-stage temporal downsampling
  +-----+----+
        |
  +-----v-------------------+
  | ASO                     |  <- data-driven spectral mask + soft physical prior
  | (Adaptive Spectral      |     of fault characteristic frequencies and harmonics
  |  Operator)              |     (frequency axis normalised by the pooled rate)
  +-----+-------------------+
        |
  +-----v----+
  | AvgPool  |  <- post-stage temporal downsampling
  +-----+----+
        |
  +-----v-------------------+
  | SSM-Attention block     |  <- S4D-Real recursion + multi-head attention,
  |                         |     fused by a learnable gate
  +-----+-------------------+
        |
  +-----v----+
  |   GAP    |  <- global average pooling
  +-----+----+
        |
  +-----v-------------------+
  | Segmented prototype     |  <- 128-d embedding split into 4 x 32-d segments,
  | classifier              |     dynamically weighted cosine similarity
  +-------------------------+
```

### Key Components

| Module | Description |
|--------|-------------|
| **Learnable Morlet frontend** | Parameterised complex Morlet filter bank embedded as the first network layer; each filter has an explicit centre frequency and a log-domain scale constraint, both reportable after training. |
| **ASO** | Amplitude-guided adaptive spectral operator. A dual-path mask adds a data-driven spectral mask to a soft physical prior built from the fault characteristic frequencies and their harmonics (Gaussian kernels on the digital frequency axis). |
| **SSM-Attention block** | Single-layer dual-path module pairing an S4D-Real state-space path with a multi-head attention path, combined through a learnable fusion gate. |
| **Segmented prototype classifier** | Partitions the 128-d embedding into four 32-d segments, computes per-segment prototypes and a dynamically weighted cosine similarity. |

## Project Structure

```
PIAS_SSM/                       # Project root
+-- pias_ssm/                   # Python package
|   +-- __init__.py             # Public API
|   +-- config.py               # Typed experiment configuration
|   +-- models/
|   |   +-- model.py            # PIASSSM (main model)
|   |   +-- wavelet.py          # MorletFrontend, ConvolutionFrontend
|   |   +-- spectral.py         # AdaptiveSpectralOperator, SimpleSpectralOperator
|   |   +-- state_space.py      # SSMAttentionBlock, S4D
|   |   +-- classifier.py       # SegmentedPrototypeClassifier
|   +-- data/
|   |   +-- dataset.py          # BearingDataset, episode sampling
|   |   +-- noise.py            # Noise injection (Gaussian / colored / impulsive / ...)
|   +-- training/
|   |   +-- engine.py           # Training loop, checkpoints
|   |   +-- evaluator.py        # Few-shot evaluation
|   |   +-- statistics.py       # Bootstrap CIs, paired tests
|   +-- utils/
|       +-- io.py               # Atomic JSON writing
|       +-- seed.py             # set_seed
+-- scripts/                    # Data preprocessing
|   +-- preprocess_cwru.py      # CWRU dataset
|   +-- preprocess_ottawa.py    # University of Ottawa dataset
+-- configs/                    # Experiment configurations
|   +-- cwru.json
|   +-- ottawa.json
+-- train.py                    # Training entry point
+-- evaluate.py                 # Evaluation entry point
+-- requirements.txt
+-- LICENSE
+-- README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Data Preprocessing

```bash
python scripts/preprocess_cwru.py \
    --data-dir   /path/to/CWRU/12k\ Drive\ End\ Bearing\ Fault\ Data \
    --normal-dir /path/to/CWRU/Normal\ Baseline\ Data \
    --output     data/cwru_workbuddy_10way.pt

python scripts/preprocess_ottawa.py \
    --raw-dir /path/to/Ottawa \
    --output  data/ottawa_workbuddy.pt
```

The generated `.pt` files are read from `data/`; the tensor format is documented in
`data/README.md`.

## Usage

```bash
# Train
python train.py --config configs/cwru.json
python train.py --config configs/ottawa.json

# Evaluate a checkpoint at a given SNR
python evaluate.py --config configs/cwru.json \
    --checkpoint outputs/CWRU/main/seed_42/checkpoint_last.pt --snr -5

# Evaluate under a specific noise type
python evaluate.py --config configs/cwru.json \
    --checkpoint outputs/CWRU/main/seed_42/checkpoint_last.pt \
    --snr -5 --noise impulsive
```

Supported `--noise` values: `none`, `gaussian`, `pink`, `brown`, `impulsive`,
`nonstationary`, `narrowband`.

Training writes per-seed checkpoints, the training history, and the full SNR sweep under
`outputs/<dataset>/<experiment_name>/seed_<seed>/`.

## Configuration

All experiment settings live in a JSON file under `configs/`, parsed by
`pias_ssm/config.py`. The main blocks are `model` (architecture and hyper-parameters),
`episode` (N-way K-shot protocol and split), and `training` (optimiser, schedule,
curriculum, seeds), plus `evaluation_snrs`.

## Citation

If you use this code in your research, please cite the paper.

## License

MIT License. See `LICENSE` for details.
