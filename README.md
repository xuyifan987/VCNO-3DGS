# VCNO-3DGS: Vehicle-Context Neural Optimization for 3D Gaussian Splatting

VCNO-3DGS is a vehicle-oriented 3D Gaussian Splatting framework for object-centric
reconstruction under sparse-view and degraded-view conditions. It extends 2D
Gaussian Splatting with reliability-aware appearance supervision, adaptive
Gaussian allocation, and confidence-gated TSDF geometry refinement.

This repository contains the training, rendering, evaluation, and CUDA extension
code used by VCNO-3DGS. Datasets and trained outputs are not included.

## How to Setup

Our experiments were run with CUDA-capable PyTorch on Linux/Windows workstations.
Create the conda environment with:

```bash
git clone https://github.com/xuyifan987/VCNO-3DGS.git
cd VCNO-3DGS
conda env create -f environment.yml
conda activate vcno_3dgs
```

The environment installs the local rasterization, KNN, and inverse-projection
extensions listed in `environment.yml`.

## Dataset

VCNO-3DGS uses the same COLMAP-style scene format as 3DGS/2DGS:

```text
<scene>
|-- images
`-- sparse/0
    |-- cameras.bin
    |-- images.bin
    `-- points3D.bin
```

Use `--eval` when a held-out test split is needed for evaluation.

## How to Train

Train the full VCNO-3DGS model with:

```bash
python train.py -s <path-to-scene> -m output/<scene>/vcno --eval --vehicle_full
```

The flag `--vehicle_full` enables the appearance, allocation, and geometry
modules. Individual modules can be enabled with:

```bash
--vehicle_appearance
--vehicle_allocation
--vehicle_geometry
```

To train the 2DGS baseline in the same codebase, omit the VCNO flags:

```bash
python train.py -s <path-to-scene> -m output/<scene>/2dgs --eval
```

## Evaluation

Render the trained model:

```bash
python render.py -s <path-to-scene> -m output/<scene>/vcno --skip_mesh
```

Compute PSNR, SSIM, and LPIPS:

```bash
python metrics.py -m output/<scene>/vcno
```

For all-view evaluation, render both train and test views and aggregate the
streaming metrics:

```bash
python metrics_stream.py -m output/<scene>/vcno --split train
python metrics_stream.py -m output/<scene>/vcno --split test
python scripts/aggregate_all_metrics.py output/<scene>
```

## How to View

The viewer follows the 2DGS/3DGS viewer workflow. After training, open the saved
model with a compatible SIBR Gaussian viewer:

```bash
./SIBR_gaussianViewer_app -m <path-to-output>
```

## Notes

Detailed experimental settings and ablations are described in the paper. This
release keeps the public README concise and leaves paper-specific scene settings
to the manuscript and command-line defaults.

See `NOTICE.md` for the distinction between VCNO-3DGS additions and retained
upstream components.

## Acknowledgements

This project builds on:

- 3D Gaussian Splatting: https://github.com/graphdeco-inria/gaussian-splatting
- 2D Gaussian Splatting: https://github.com/hbb1/2d-gaussian-splatting

## License

This repository is released for non-commercial research and evaluation under the
upstream Gaussian Splatting license in `LICENSE.md`. Retained upstream copyright
notices are preserved in source files for license compliance.

## Citation

Citation metadata will be added after manuscript release.
