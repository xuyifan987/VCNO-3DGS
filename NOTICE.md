# Notice

VCNO-3DGS is a research codebase derived from 2D Gaussian Splatting and related
3D Gaussian Splatting components. The repository contains both upstream code and
VCNO-specific modifications.

## VCNO-3DGS Contributions

The VCNO-specific implementation includes:

- reliability-filtered vehicle-detail evidence construction;
- vehicle-context neural appearance adaptation;
- reliability- and visibility-guided primitive allocation;
- confidence-gated TSDF fusion and Gaussian surface pull;
- vehicle-focused training presets and evaluation utilities.

These additions are implemented mainly in:

```text
arguments/__init__.py
train.py
frequency_detection.py
utils/car_fusion_utils.py
utils/grid_sdf.py
utils/ngs_tsdf_prior.py
scene/gaussian_model.py
scripts/
```

## Upstream Code

Several files retain copyright and attribution notices from their original
authors, including Inria/GRAPHDECO, 2DGS, Google, PlenOctree, and ShanghaiTech
components. These notices are intentionally preserved for license compliance and
must not be replaced by VCNO-3DGS authorship notices.

The root `LICENSE.md` follows the upstream Gaussian Splatting non-commercial
research license. Users should review it before redistribution or use.
