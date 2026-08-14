#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#

from argparse import ArgumentParser
from pathlib import Path
import json
import os

from PIL import Image
import torch
import torchvision.transforms.functional as tf
from tqdm import tqdm

from lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.loss_utils import ssim


def load_image(path: Path):
    image = Image.open(path)
    return tf.to_tensor(image).unsqueeze(0)[:, :3, :, :].cuda()


def evaluate_split(scene_dir: Path, split: str, skip_lpips: bool):
    split_dir = scene_dir / split
    if not split_dir.exists():
        print(f"Skip missing split: {split_dir}")
        return None, None

    scene_results = {}
    scene_per_view = {}
    for method in sorted(os.listdir(split_dir)):
        method_dir = split_dir / method
        gt_dir = method_dir / "gt"
        renders_dir = method_dir / "renders"
        if not gt_dir.exists() or not renders_dir.exists():
            continue

        names = sorted(
            fname for fname in os.listdir(renders_dir)
            if (gt_dir / fname).exists()
        )
        if not names:
            continue

        ssims = []
        psnrs = []
        lpipss = []
        for name in tqdm(names, desc=f"{scene_dir.name}/{split}/{method}"):
            render = load_image(renders_dir / name)
            gt = load_image(gt_dir / name)
            ssims.append(ssim(render, gt).item())
            psnrs.append(psnr(render, gt).item())
            if not skip_lpips:
                lpipss.append(lpips(render, gt, net_type="vgg").item())
            del render, gt

        mean_ssim = float(torch.tensor(ssims).mean().item())
        mean_psnr = float(torch.tensor(psnrs).mean().item())
        mean_lpips = None if skip_lpips else float(torch.tensor(lpipss).mean().item())
        print(f"Scene: {scene_dir}")
        print(f"Split: {split}  Method: {method}")
        print(f"  SSIM : {mean_ssim:>12.7f}")
        print(f"  PSNR : {mean_psnr:>12.7f}")
        if mean_lpips is not None:
            print(f"  LPIPS: {mean_lpips:>12.7f}")
        print("")

        scene_results[method] = {
            "SSIM": mean_ssim,
            "PSNR": mean_psnr,
        }
        if mean_lpips is not None:
            scene_results[method]["LPIPS"] = mean_lpips
        scene_per_view[method] = {
            "SSIM": dict(zip(names, ssims)),
            "PSNR": dict(zip(names, psnrs)),
        }
        if not skip_lpips:
            scene_per_view[method]["LPIPS"] = dict(zip(names, lpipss))

    return scene_results, scene_per_view


def evaluate(model_paths, split, skip_lpips):
    for model_path in model_paths:
        scene_dir = Path(model_path)
        results, per_view = evaluate_split(scene_dir, split, skip_lpips)
        if not results:
            print("Unable to compute metrics for model", scene_dir)
            continue
        with open(scene_dir / f"results_{split}.json", "w") as fp:
            json.dump(results, fp, indent=True)
        with open(scene_dir / f"per_view_{split}.json", "w") as fp:
            json.dump(per_view, fp, indent=True)


if __name__ == "__main__":
    torch.cuda.set_device(torch.device("cuda:0"))
    parser = ArgumentParser(description="Streaming metric evaluation")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=str)
    parser.add_argument("--split", default="test", choices=["train", "test"], type=str)
    parser.add_argument("--skip_lpips", action="store_true")
    args = parser.parse_args()
    evaluate(args.model_paths, args.split, args.skip_lpips)
