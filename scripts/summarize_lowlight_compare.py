import argparse
import csv
import json
from pathlib import Path


METHODS = [
    ("2DGS", "01_2dgs"),
    ("3D-Gabor", "02_3d_gabor"),
    ("Ours", "ours_best_d020_b10_t00015"),
]


def read_result(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf8") as f:
        data = json.load(f)
    if not data:
        return None
    key = sorted(data.keys())[-1]
    metrics = data[key]
    return {
        "SSIM": float(metrics.get("SSIM", "nan")),
        "PSNR": float(metrics.get("PSNR", "nan")),
        "LPIPS": float(metrics.get("LPIPS", "nan")),
    }


def model_size_mb(model_dir: Path, iteration: int):
    ply = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply.exists():
        return None
    return ply.stat().st_size / (1024 * 1024)


def row_for(scene: str, method_label: str, model_dir: Path, iteration: int):
    metrics = read_result(model_dir / "results_all.json")
    if metrics is None:
        metrics = read_result(model_dir / "results_test.json")
    if metrics is None:
        metrics = read_result(model_dir / "results.json")
    if metrics is None:
        return None
    return {
        "Scene": scene,
        "Method": method_label,
        "SSIM": metrics["SSIM"],
        "PSNR": metrics["PSNR"],
        "LPIPS": metrics["LPIPS"],
        "MemMB": model_size_mb(model_dir, iteration),
        "Path": str(model_dir),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--ours-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=20000)
    args = parser.parse_args()

    scenes = sorted(
        set(p.name for p in args.baseline_root.iterdir() if p.is_dir())
        | set(p.name for p in args.ours_root.iterdir() if p.is_dir())
    )

    rows = []
    for scene in scenes:
        for label, subdir in METHODS:
            root = args.ours_root if label == "Ours" else args.baseline_root
            model_dir = root / scene / subdir
            row = row_for(scene, label, model_dir, args.iteration)
            if row is not None:
                rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Scene", "Method", "SSIM", "PSNR", "LPIPS", "MemMB", "Path"]
    with args.out.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md = args.out.with_suffix(".md")
    with md.open("w", encoding="utf8") as f:
        f.write("| Scene | Method | SSIM↑ | PSNR↑ | LPIPS↓ | Mem |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for row in rows:
            mem = "" if row["MemMB"] is None else f"{row['MemMB']:.2f}MB"
            f.write(
                f"| {row['Scene']} | {row['Method']} | "
                f"{row['SSIM']:.4f} | {row['PSNR']:.2f} | {row['LPIPS']:.4f} | {mem} |\n"
            )

    print(f"Wrote {args.out}")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
