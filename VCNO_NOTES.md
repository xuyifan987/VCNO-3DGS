# VCNO-3DGS Implementation Notes

This repository implements VCNO-3DGS on top of the 2D Gaussian Splatting
training pipeline. The public interface is organized around three paper-level
module switches:

```bash
--vehicle_appearance
--vehicle_allocation
--vehicle_geometry
--vehicle_full
```

`--vehicle_full` enables all three modules.

## Module Mapping

| Paper module | Public switch | Main low-level implementation flags |
|---|---|---|
| Vehicle-context neural appearance | `--vehicle_appearance` | `car_fusion_adaptive_gabor`, detail photometric/SSIM/structure losses |
| Reliability-guided primitive allocation | `--vehicle_allocation` | detail-routed densification, visibility-aware densification, reliability floor schedule |
| Confidence-gated geometry refinement | `--vehicle_geometry` | scheduled TSDF fusion and confidence-gated TSDF pull |

The older `--ours_*` aliases are kept for backward compatibility:

```text
--ours_appearance -> --vehicle_appearance
--ours_frequency  -> --vehicle_allocation
--ours_geometry   -> --vehicle_geometry
--ours_full       -> --vehicle_full
```

The internal `car_fusion_*` option names are retained because trained model
configuration files (`cfg_args`) and old ablation logs may still contain them.
They should be treated as low-level implementation controls rather than the
paper-facing method name.

## Default Training Entry Points

Baseline 2DGS-style run:

```bash
python train.py -s <dataset> -m <output> --eval
```

Full VCNO-3DGS run:

```bash
python train.py -s <dataset> -m <output> --eval --vehicle_full
```

Typical paper-style caps are set from the command line:

```bash
python train.py \
  -s <dataset> \
  -m <output> \
  --eval \
  --vehicle_full \
  --iterations 20000 \
  --init_point_num <init-points> \
  --max_primitive_num <primitive-cap> \
  --test_iterations 20000 \
  --save_iterations 20000
```

## Design Summary

VCNO-3DGS builds a reliability-filtered vehicle-detail evidence field from
rendering residuals, alpha visibility, local frequency/detail response, and
vehicle-context confidence. The same evidence drives:

1. Reliable detail supervision and neural appearance adaptation.
2. Primitive allocation under a scene-specific hard cap.
3. TSDF-based surface refinement guarded by image and surface confidence.

This shared evidence field is the main distinction from using independent
auxiliary losses or a generic densification schedule.
