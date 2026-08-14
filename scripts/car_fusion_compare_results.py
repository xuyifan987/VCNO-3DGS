import argparse
import csv
import json
import math
from pathlib import Path


def load_result(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return None
    method, metrics = next(iter(data.items()))
    return {
        "path": str(path),
        "scene": path.parent.parent.name,
        "run": path.parent.name,
        "method": method,
        "psnr": float(metrics.get("PSNR", "nan")),
        "ssim": float(metrics.get("SSIM", "nan")),
        "lpips": float(metrics.get("LPIPS", "nan")),
    }


def choose_baselines(rows, baseline_token):
    baselines = {}
    for row in rows:
        if baseline_token.lower() in row["run"].lower():
            baselines.setdefault(row["scene"], row)
    return baselines


def metric_pass(delta, target, lower_is_better=False):
    if delta == "" or delta is None:
        return None
    if isinstance(delta, float) and math.isnan(delta):
        return None
    return delta <= target if lower_is_better else delta >= target


def shortfall(delta, target, lower_is_better=False):
    if delta == "" or delta is None:
        return ""
    if isinstance(delta, float) and math.isnan(delta):
        return ""
    missing = delta - target if lower_is_better else target - delta
    return max(missing, 0.0)


def next_focus(row):
    gaps = []
    if row.get("missing_psnr") not in ("", None):
        gaps.append(("PSNR", float(row["missing_psnr"])))
    if row.get("missing_ssim") not in ("", None):
        gaps.append(("SSIM", float(row["missing_ssim"])))
    if row.get("missing_lpips") not in ("", None):
        gaps.append(("LPIPS", float(row["missing_lpips"])))
    gaps = [(name, gap) for name, gap in gaps if gap > 0.0]
    if not gaps:
        return "target_met"
    gaps.sort(key=lambda item: item[1], reverse=True)
    return gaps[0][0].lower()


def suggestion_for_focus(focus):
    if focus == "psnr":
        return "raise detail_photometric_weight or metric_adaptive_max_boost; keep residual_floor small"
    if focus == "ssim":
        return "raise contrast/detail_ssim/structure weights; inspect edge_boost and reliability gate"
    if focus == "lpips":
        return "raise adaptive_gabor detail routing, residual_detail_power, or TSDF confidence blend"
    if focus == "target_met":
        return "hold config and repeat with another seed/scene"
    return ""


def main():
    parser = argparse.ArgumentParser(description="Summarize car-fusion metric deltas from results.json files.")
    parser.add_argument("roots", nargs="+", help="Scene/output roots to scan recursively.")
    parser.add_argument("--baseline-token", default="baseline", help="Run-name token used as per-scene baseline.")
    parser.add_argument("--target-psnr", type=float, default=0.0, help="Minimum PSNR delta marked as pass.")
    parser.add_argument("--target-ssim", type=float, default=0.0, help="Minimum SSIM delta marked as pass.")
    parser.add_argument("--target-lpips", type=float, default=0.0, help="Maximum LPIPS delta marked as pass; negative means improvement.")
    parser.add_argument("--ignore-missing-lpips", action="store_true", help="Do not fail rows whose LPIPS value is missing or NaN.")
    parser.add_argument("--csv", default="", help="Optional CSV output path.")
    args = parser.parse_args()

    rows = []
    for root in args.roots:
        for path in Path(root).rglob("results.json"):
            row = load_result(path)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda x: (x["scene"], x["run"]))

    baselines = choose_baselines(rows, args.baseline_token)
    for row in rows:
        base = baselines.get(row["scene"])
        if base is None:
            row["d_psnr"] = ""
            row["d_ssim"] = ""
            row["d_lpips"] = ""
            row["missing_psnr"] = ""
            row["missing_ssim"] = ""
            row["missing_lpips"] = ""
            row["target"] = ""
            row["focus"] = ""
            row["suggestion"] = ""
        else:
            row["d_psnr"] = row["psnr"] - base["psnr"]
            row["d_ssim"] = row["ssim"] - base["ssim"]
            row["d_lpips"] = row["lpips"] - base["lpips"]
            row["missing_psnr"] = shortfall(row["d_psnr"], args.target_psnr)
            row["missing_ssim"] = shortfall(row["d_ssim"], args.target_ssim)
            row["missing_lpips"] = shortfall(row["d_lpips"], args.target_lpips, lower_is_better=True)
            if row is base:
                row["target"] = "BASE"
                row["focus"] = "baseline"
                row["suggestion"] = ""
            else:
                pass_psnr = metric_pass(row["d_psnr"], args.target_psnr)
                pass_ssim = metric_pass(row["d_ssim"], args.target_ssim)
                pass_lpips = metric_pass(row["d_lpips"], args.target_lpips, lower_is_better=True)
                if pass_lpips is None and args.ignore_missing_lpips:
                    pass_lpips = True
                row["target"] = "PASS" if pass_psnr and pass_ssim and pass_lpips else "MISS"
                row["focus"] = next_focus(row)
                row["suggestion"] = suggestion_for_focus(row["focus"])

    fields = [
        "scene", "run", "method", "psnr", "ssim", "lpips",
        "d_psnr", "d_ssim", "d_lpips",
        "missing_psnr", "missing_ssim", "missing_lpips",
        "target", "focus", "suggestion"
    ]
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: row[k] for k in fields} for row in rows)

    print(",".join(fields))
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append(str(value))
        print(",".join(values))


if __name__ == "__main__":
    main()
