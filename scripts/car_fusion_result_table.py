import argparse
import json
from pathlib import Path


def read_result(path):
    with path.open("r", encoding="utf8") as f:
        data = json.load(f)
    if not data:
        return None
    key = sorted(data.keys())[-1]
    metrics = data[key]
    return key, float(metrics["PSNR"]), float(metrics["SSIM"]), float(metrics["LPIPS"])


def scene_name(model_dir):
    parts = model_dir.parts
    for i, part in enumerate(parts):
        if part.lower().startswith("car") and i + 1 < len(parts):
            return part
    return model_dir.parent.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--baseline-token", default="baseline")
    args = parser.parse_args()

    rows = []
    seen = set()
    for root in args.roots:
        result_paths = []
        if root.is_file():
            result_paths.append(root)
        elif root.is_dir():
            result_paths.extend(root.rglob("results.json"))
            result_paths.extend(root.rglob("results_test.json"))
        for result_path in sorted(result_paths):
            resolved = result_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            item = read_result(result_path)
            if item is None:
                continue
            iteration_key, psnr, ssim, lpips = item
            model_dir = result_path.parent
            rows.append(
                {
                    "scene": scene_name(model_dir),
                    "name": model_dir.name,
                    "iter": iteration_key,
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips": lpips,
                    "path": str(model_dir),
                }
            )

    baselines = {}
    for row in rows:
        if args.baseline_token.lower() in row["name"].lower():
            baselines.setdefault(row["scene"], row)

    print("scene,name,iter,psnr,d_psnr,ssim,d_ssim,lpips,d_lpips,path")
    for row in sorted(rows, key=lambda r: (r["scene"], r["name"])):
        base = baselines.get(row["scene"])
        if base is None:
            d_psnr = d_ssim = d_lpips = ""
        else:
            d_psnr = f"{row['psnr'] - base['psnr']:.6f}"
            d_ssim = f"{row['ssim'] - base['ssim']:.6f}"
            d_lpips = f"{row['lpips'] - base['lpips']:.6f}"
        print(
            f"{row['scene']},{row['name']},{row['iter']},"
            f"{row['psnr']:.6f},{d_psnr},"
            f"{row['ssim']:.6f},{d_ssim},"
            f"{row['lpips']:.6f},{d_lpips},"
            f"{row['path']}"
        )


if __name__ == "__main__":
    main()
