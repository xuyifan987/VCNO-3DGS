import argparse
import csv
import json
from pathlib import Path


METRICS = ("SSIM", "PSNR", "LPIPS")


def _read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf8") as f:
        return json.load(f)


def _method_key(per_view):
    if not per_view:
        return None
    return sorted(per_view.keys())[-1]


def _collect_values(model_dir: Path):
    values = {metric: [] for metric in METRICS}
    counts = {"train": 0, "test": 0}
    for split in ("train", "test"):
        per_view = _read_json(model_dir / f"per_view_{split}.json")
        key = _method_key(per_view)
        if key is None:
            continue
        split_metrics = per_view[key]
        split_count = 0
        for metric in METRICS:
            metric_values = split_metrics.get(metric, {})
            vals = [float(v) for v in metric_values.values()]
            values[metric].extend(vals)
            split_count = max(split_count, len(vals))
        counts[split] = split_count
    return values, counts


def _model_size_mb(model_dir: Path, iteration: int):
    ply = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply.exists():
        return None
    return ply.stat().st_size / (1024 * 1024)


def summarize(root: Path, iteration: int):
    rows = []
    for model_dir in sorted(p for p in root.rglob("*") if p.is_dir()):
        if not (model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply").exists():
            continue
        values, counts = _collect_values(model_dir)
        if not values["PSNR"]:
            continue
        scene = model_dir.parent.name
        row = {
            "scene": scene,
            "config": model_dir.name,
            "train_views": counts["train"],
            "test_views": counts["test"],
            "all_views": counts["train"] + counts["test"],
            "MemMB": _model_size_mb(model_dir, iteration),
        }
        all_result = {}
        for metric in METRICS:
            vals = values[metric]
            if vals:
                row[metric] = sum(vals) / len(vals)
                all_result[metric] = row[metric]
        with (model_dir / "results_all.json").open("w", encoding="utf8") as f:
            json.dump({f"ours_{iteration}": all_result}, f, indent=True)
        rows.append(row)
    return rows


def write_outputs(rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ablation_all_summary.csv"
    fields = ["scene", "config", "SSIM", "PSNR", "LPIPS", "MemMB", "train_views", "test_views", "all_views"]
    with csv_path.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    md_path = out_dir / "ablation_all_summary.md"
    with md_path.open("w", encoding="utf8") as f:
        f.write("| Scene | Configuration | SSIM↑ | PSNR↑ | LPIPS↓ | Mem |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['scene']} | {row['config']} | "
                f"{row.get('SSIM', float('nan')):.4f} | "
                f"{row.get('PSNR', float('nan')):.2f} | "
                f"{row.get('LPIPS', float('nan')):.4f} | "
                f"{row.get('MemMB', float('nan')):.2f}MB |\n"
            )
    return csv_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Aggregate train/test metrics into all-view metrics.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--iteration", type=int, default=20000)
    args = parser.parse_args()
    rows = summarize(args.root, args.iteration)
    csv_path, md_path = write_outputs(rows, args.root)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
