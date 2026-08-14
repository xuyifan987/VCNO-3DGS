from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ModuleNotFoundError:
    EventAccumulator = None


RUN_DIR: Path | None = None
OUT_DIR = Path("mechanism_curves")


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": True,
        "legend.framealpha": 0.68,
        "legend.edgecolor": "0.85",
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)


def load_events(run_dir: Path) -> EventAccumulator:
    if EventAccumulator is None:
        raise ModuleNotFoundError("tensorboard")
    event_files = sorted(run_dir.glob("events.out.tfevents*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file found in {run_dir}")
    acc = EventAccumulator(
        str(event_files[0]),
        size_guidance={
            "scalars": 0,
            "images": 0,
            "histograms": 0,
            "compressedHistograms": 0,
            "tensors": 0,
        },
    )
    acc.Reload()
    return acc


def scalar(acc: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
    values = acc.Scalars(tag)
    if not values:
        raise KeyError(tag)
    x = np.array([v.step for v in values], dtype=np.float64)
    y = np.array([v.value for v in values], dtype=np.float64)
    return x, y


def smooth(y: np.ndarray, window: int = 31) -> np.ndarray:
    if y.size < 3 or window <= 1:
        return y
    window = int(min(window, y.size))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return y
    pad = window // 2
    padded = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def try_scalar(acc: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        return scalar(acc, tag)
    except KeyError:
        return None


def save_source_csv(path: Path, series: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["series", "iteration", "value"])
        for name, (x, y) in series.items():
            for xi, yi in zip(x, y):
                writer.writerow([name, int(xi), float(yi)])


def load_source_csv(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    values: dict[str, list[tuple[float, float]]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.setdefault(row["series"], []).append((float(row["iteration"]), float(row["value"])))
    return {
        name: (
            np.array([v[0] for v in rows], dtype=np.float64),
            np.array([v[1] for v in rows], dtype=np.float64),
        )
        for name, rows in values.items()
    }


def style_axes(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.6, alpha=0.6)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.4, alpha=0.35)
    ax.minorticks_on()


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in {
        "png": {"dpi": 600},
        "pdf": {},
        "svg": {},
    }.items():
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", **kwargs)


def add_overlay_legend(ax: plt.Axes, lines: list, labels: list[str], **kwargs) -> None:
    legend = ax.legend(
        lines,
        labels,
        frameon=True,
        facecolor="white",
        framealpha=0.68,
        edgecolor="0.82",
        fancybox=True,
        **kwargs,
    )
    legend.set_zorder(20)
    return None


def plot_appearance(acc: EventAccumulator) -> None:
    projected = scalar(acc, "vehicle_appearance/projected_detail")
    confidence = scalar(acc, "vehicle_appearance/confidence")
    detail_photo = scalar(acc, "car_fusion/detail_photometric_loss")
    detail_ssim = scalar(acc, "car_fusion/detail_ssim_loss")

    # Detail losses are logged at the training-report interval; combine them as the
    # optimization signal that drives the appearance branch.
    n = min(len(detail_photo[0]), len(detail_ssim[0]))
    detail_loss = (detail_photo[0][:n], detail_photo[1][:n] + detail_ssim[1][:n])
    projected_s = (projected[0], smooth(projected[1], 41))
    confidence_s = (confidence[0], smooth(confidence[1], 41))
    detail_loss_s = (detail_loss[0], smooth(detail_loss[1], 41))

    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    ax.plot(projected_s[0], projected_s[1], color="#1f55ff", linewidth=1.45, label="Projected detail")
    ax.plot(confidence_s[0], confidence_s[1], color="#d7191c", linewidth=1.35, label="Appearance confidence")
    style_axes(ax, "Training iterations", "Appearance signal")
    ax.set_ylim(0.0, max(1.0, np.nanmax(confidence[1]) * 1.05))

    ax2 = ax.twinx()
    ax2.plot(detail_loss_s[0], detail_loss_s[1], color="#d7191c", linewidth=1.25, linestyle="--", label="Reliable detail loss")
    ax2.set_ylabel("Detail loss")
    ax2.spines["top"].set_visible(False)

    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    add_overlay_legend(ax, lines, labels, loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    save_source_csv(
        OUT_DIR / "appearance_source.csv",
        {
            "projected_detail_smoothed": projected_s,
            "appearance_confidence_smoothed": confidence_s,
            "reliable_detail_loss_smoothed": detail_loss_s,
        },
    )
    save_figure(fig, "fig_appearance_mechanism")
    plt.close(fig)


def plot_appearance_from_csv() -> None:
    data = load_source_csv(OUT_DIR / "appearance_source.csv")
    projected_s = data["projected_detail_smoothed"]
    confidence_s = data["appearance_confidence_smoothed"]
    detail_loss_s = data["reliable_detail_loss_smoothed"]

    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    ax.plot(projected_s[0], projected_s[1], color="#1f55ff", linewidth=1.45, label="Projected detail")
    ax.plot(confidence_s[0], confidence_s[1], color="#d7191c", linewidth=1.35, label="Appearance confidence")
    style_axes(ax, "Training iterations", "Appearance signal")
    ax.set_ylim(0.0, max(1.0, np.nanmax(confidence_s[1]) * 1.05))

    ax2 = ax.twinx()
    ax2.plot(detail_loss_s[0], detail_loss_s[1], color="#d7191c", linewidth=1.25, linestyle="--", label="Reliable detail loss")
    ax2.set_ylabel("Detail loss")
    ax2.spines["top"].set_visible(False)

    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    add_overlay_legend(ax, lines, labels, loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    save_figure(fig, "fig_appearance_mechanism")
    plt.close(fig)


def plot_budget(acc: EventAccumulator) -> None:
    total_points = scalar(acc, "total_points")
    after_densify = try_scalar(acc, "total_points_after_densify")
    view_support = scalar(acc, "car_fusion/densify_multiview_confidence_mean")
    reliability_floor = scalar(acc, "car_fusion/densify_reliability_floor")
    total_points_s = (total_points[0], smooth(total_points[1], 11))
    after_densify_s = (after_densify[0], smooth(after_densify[1], 11)) if after_densify is not None else None

    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    ax.plot(total_points_s[0], total_points_s[1], color="#1f55ff", linewidth=1.45, label="Total Gaussians")
    if after_densify_s is not None:
        ax.plot(after_densify_s[0], after_densify_s[1], color="#d7191c", linewidth=1.25, label="After budget update")
    ax.axhline(20000, color="0.25", linewidth=1.0, linestyle=":", label="Primitive budget")
    style_axes(ax, "Training iterations", "Number of Gaussians")
    ax.ticklabel_format(axis="y", style="plain")

    ax2 = ax.twinx()
    ax2.plot(view_support[0], smooth(view_support[1], 9), color="#1f55ff", linewidth=1.25, linestyle="--", label="Multi-view confidence")
    ax2.plot(reliability_floor[0], smooth(reliability_floor[1], 9), color="#d7191c", linewidth=1.25, linestyle="--", label="Reliability floor")
    ax2.set_ylabel("Budgeting confidence")
    ax2.set_ylim(0.0, 1.0)
    ax2.spines["top"].set_visible(False)

    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    add_overlay_legend(ax, lines, labels, loc="lower right", fontsize=8)
    fig.tight_layout()
    save_source_csv(
        OUT_DIR / "budget_source.csv",
        {
            "total_gaussians_smoothed": total_points_s,
            "after_budget_update_smoothed": after_densify_s if after_densify_s is not None else total_points_s,
            "multiview_confidence_smoothed": (view_support[0], smooth(view_support[1], 9)),
            "reliability_floor_smoothed": (reliability_floor[0], smooth(reliability_floor[1], 9)),
        },
    )
    save_figure(fig, "fig_budget_mechanism")
    plt.close(fig)


def plot_budget_from_csv() -> None:
    data = load_source_csv(OUT_DIR / "budget_source.csv")
    total_points_s = data["total_gaussians_smoothed"]
    after_densify_s = data["after_budget_update_smoothed"]
    view_support_s = data["multiview_confidence_smoothed"]
    reliability_floor_s = data["reliability_floor_smoothed"]

    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    ax.plot(total_points_s[0], total_points_s[1], color="#1f55ff", linewidth=1.45, label="Total Gaussians")
    ax.plot(after_densify_s[0], after_densify_s[1], color="#d7191c", linewidth=1.25, label="After budget update")
    ax.axhline(20000, color="0.25", linewidth=1.0, linestyle=":", label="Primitive budget")
    style_axes(ax, "Training iterations", "Number of Gaussians")
    ax.ticklabel_format(axis="y", style="plain")

    ax2 = ax.twinx()
    ax2.plot(view_support_s[0], view_support_s[1], color="#1f55ff", linewidth=1.25, linestyle="--", label="Multi-view confidence")
    ax2.plot(reliability_floor_s[0], reliability_floor_s[1], color="#d7191c", linewidth=1.25, linestyle="--", label="Reliability floor")
    ax2.set_ylabel("Budgeting confidence")
    ax2.set_ylim(0.0, 1.0)
    ax2.spines["top"].set_visible(False)

    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    add_overlay_legend(ax, lines, labels, loc="lower right", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "fig_budget_mechanism")
    plt.close(fig)


def plot_tsdf(acc: EventAccumulator) -> None:
    mean_abs_sdf = scalar(acc, "car_fusion/tsdf_mean_abs_sdf")
    pull_gate = scalar(acc, "car_fusion/tsdf_pull_gate")
    pulled = scalar(acc, "car_fusion/tsdf_pulled")
    surface_points = scalar(acc, "car_fusion/tsdf_surface_points")

    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    ax.plot(mean_abs_sdf[0], mean_abs_sdf[1], color="#1f55ff", linewidth=1.35, label="Mean |SDF|")
    ax.plot(pull_gate[0], pull_gate[1], color="#d7191c", linewidth=1.25, label="Confidence pull gate")
    style_axes(ax, "Training iterations", "TSDF alignment signal")

    ax2 = ax.twinx()
    ax2.plot(pulled[0], pulled[1], color="#1f55ff", linewidth=1.25, linestyle="--", label="Pulled Gaussians")
    ax2.plot(surface_points[0], surface_points[1], color="#d7191c", linewidth=1.25, linestyle="--", label="Surface samples")
    ax2.set_ylabel("Gaussians / surface samples")
    ax2.spines["top"].set_visible(False)

    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    add_overlay_legend(ax, lines, labels, loc="upper right", fontsize=8)
    fig.tight_layout()
    save_source_csv(
        OUT_DIR / "tsdf_source.csv",
        {
            "mean_abs_sdf": mean_abs_sdf,
            "pull_gate": pull_gate,
            "pulled_gaussians": pulled,
            "surface_samples": surface_points,
        },
    )
    save_figure(fig, "fig_tsdf_mechanism")
    plt.close(fig)


def plot_tsdf_from_csv() -> None:
    data = load_source_csv(OUT_DIR / "tsdf_source.csv")
    mean_abs_sdf = data["mean_abs_sdf"]
    pull_gate = data["pull_gate"]
    pulled = data["pulled_gaussians"]
    surface_points = data["surface_samples"]

    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    ax.plot(mean_abs_sdf[0], mean_abs_sdf[1], color="#1f55ff", linewidth=1.35, label="Mean |SDF|")
    ax.plot(pull_gate[0], pull_gate[1], color="#d7191c", linewidth=1.25, label="Confidence pull gate")
    style_axes(ax, "Training iterations", "TSDF alignment signal")

    ax2 = ax.twinx()
    ax2.plot(pulled[0], pulled[1], color="#1f55ff", linewidth=1.25, linestyle="--", label="Pulled Gaussians")
    ax2.plot(surface_points[0], surface_points[1], color="#d7191c", linewidth=1.25, linestyle="--", label="Surface samples")
    ax2.set_ylabel("Gaussians / surface samples")
    ax2.spines["top"].set_visible(False)

    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    add_overlay_legend(ax, lines, labels, loc="upper right", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "fig_tsdf_mechanism")
    plt.close(fig)


def plot_combined() -> None:
    images = [
        OUT_DIR / "fig_appearance_mechanism.png",
        OUT_DIR / "fig_budget_mechanism.png",
        OUT_DIR / "fig_tsdf_mechanism.png",
    ]
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 9.6))
    for ax, img, label in zip(axes, images, ["a", "b", "c"]):
        arr = plt.imread(img)
        ax.imshow(arr)
        ax.axis("off")
        ax.text(0.005, 0.985, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    fig.tight_layout(pad=0.2)
    save_figure(fig, "fig_three_mechanism_curves")
    plt.close(fig)


def main() -> None:
    global RUN_DIR, OUT_DIR
    parser = argparse.ArgumentParser(description="Plot VCNO-3DGS mechanism curves from TensorBoard scalars or saved CSV files.")
    parser.add_argument("--run_dir", type=Path, default=None, help="Training run directory containing TensorBoard event files.")
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR, help="Directory for figures and source CSV files.")
    args = parser.parse_args()

    RUN_DIR = args.run_dir
    OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if RUN_DIR is not None:
        acc = load_events(RUN_DIR)
        plot_appearance(acc)
        plot_budget(acc)
        plot_tsdf(acc)
    else:
        plot_appearance_from_csv()
        plot_budget_from_csv()
        plot_tsdf_from_csv()
    plot_combined()
    print(f"Saved mechanism curves to {OUT_DIR}")


if __name__ == "__main__":
    main()
