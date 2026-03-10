"""
ELIT Multi-Budget Inference Script.

Generates a single image at all possible inference budgets for an ELIT model,
saves them to a folder, and measures latency and FLOPs (after warming up the model).
Produces figures for budget vs FLOPs and budget vs latency.

Usage:
  python elit_multibudget_inference.py \
      --model ELIT-SiT-XL/2 \
      --ckpt path/to/ckpt.pt \
      --resolution 256 \
      --class-label 207 \
      --output-dir multibudget_results

  # From YAML config:
  python elit_multibudget_inference.py \
      --config experiments_updated/generation/elit_sit_b_256.yaml \
      --ckpt path/to/ckpt.pt \
      --class-label 207
"""

import os
import time
import json
import argparse

import yaml
import torch
import torch.distributed as dist
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from diffusers.models import AutoencoderKL
from torch.utils.flop_counter import FlopCounterMode

# Import model registries
from models.sit_elit import SiT_models as _elit_models
from utils import load_legacy_checkpoints


def _ensure_distributed():
    """Initialize a single-process distributed group if not already initialized.

    The ELIT masking strategy calls dist.get_rank() during training-mode
    forward passes.  Even though inference_budget != None bypasses that
    code path, we still need an initialized process group because the
    masking strategy object is created with synchronized_budget_sampling=True
    and other dist-dependent defaults.  This helper makes the script work
    with both plain `python` and `torchrun` launches.
    """
    if dist.is_initialized():
        return
    # Set minimal env vars for a single-process group when launched with
    # plain `python` (torchrun already sets these).
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ELIT Multi-Budget Inference: generate a single image at "
        "every budget, measure latency & FLOPs, and produce plots.",
    )

    # Config
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file.")

    # Model
    parser.add_argument("--model", type=str, default="ELIT-SiT-XL/2",
                        choices=list(_elit_models.keys()))
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to model checkpoint (.pt).")
    parser.add_argument("--resolution", type=int, default=256,
                        choices=[256, 512])
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action="store_true", default=False)
    parser.add_argument("--qk-norm", action="store_true", default=False)

    # REPA
    parser.add_argument("--enable-repa", action="store_true", default=False)
    parser.add_argument("--projector-embed-dims", type=str, default="768")

    # ELIT-specific
    parser.add_argument("--elit-max-mask-prob", type=float, default=0.0, help="Maximum masking probability for ELIT during training (0.0 = no masking)")
    parser.add_argument("--elit-min-mask-prob", type=float, default=None)
    parser.add_argument("--elit-group-size", type=int, default=4)

    # VAE
    parser.add_argument("--vae", type=str, default="ema", choices=["ema", "mse"])

    # Sampling
    parser.add_argument("--mode", type=str, default="ode", choices=["ode", "sde"])
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--path-type", type=str, default="linear",
                        choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action="store_true", default=False)
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=1.0)

    # Legacy
    parser.add_argument("--legacy", action="store_true", default=False)

    # Generation settings
    parser.add_argument("--class-label", type=int, default=207,
                        help="ImageNet class label for generation (default: 207 = golden retriever).")
    parser.add_argument("--seed", type=int, default=42)

    # Budgets
    parser.add_argument("--budgets", type=float, nargs="+", default=None,
                        help="List of inference budgets to evaluate. "
                             "If not set, uses np.arange(0.1, 1.05, 0.1).")

    # Benchmarking
    parser.add_argument("--warmup-iters", type=int, default=3,
                        help="Number of warmup iterations before measuring.")
    parser.add_argument("--repeat", type=int, default=5,
                        help="Number of repeated runs per budget for latency measurement.")

    # Output
    parser.add_argument("--output-dir", type=str, default="multibudget_results")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)

    # ── Parse with YAML support ──
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        # YAML values fill in only unset / default CLI values
        yaml_cfg = {k.replace("-", "_"): v for k, v in yaml_cfg.items()}
        for k, v in yaml_cfg.items():
            if hasattr(args, k) and parser.get_default(k) == getattr(args, k):
                setattr(args, k, v)

    if args.budgets is None:
        args.budgets = np.arange(0.1, 1.05, 0.1).tolist()

    return args


def build_model(args, device):
    """Build the ELIT model."""
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // 8

    enable_repa = getattr(args, "enable_repa", False)
    z_dims = None
    if enable_repa and args.projector_embed_dims:
        z_dims = [int(d) for d in args.projector_embed_dims.split(",")]

    model_kwargs = dict(
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=True,
        enable_repa=enable_repa,
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
        enable_elit=True,
        elit_max_mask_prob=args.elit_max_mask_prob,
        elit_min_mask_prob=args.elit_min_mask_prob,
        group_size=args.elit_group_size,
        **block_kwargs,
    )

    model = _elit_models[args.model](**model_kwargs).to(device)
    return model


def load_checkpoint(model, args, device):
    """Load checkpoint into model."""
    state_dict = torch.load(args.ckpt, map_location=device, weights_only=False)["ema"]
    if args.legacy:
        state_dict = load_legacy_checkpoints(
            state_dict=state_dict, encoder_depth=args.encoder_depth
        )
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"  Warning: Missing keys ({len(result.missing_keys)})")
    if result.unexpected_keys:
        print(f"  Warning: Unexpected keys ({len(result.unexpected_keys)})")
    model.eval()
    return model


@torch.no_grad()
def sample_single(model, z, y, args, device, inference_budget):
    """Run the full ODE/SDE sampling loop for a single image with a given budget."""
    from samplers import euler_sampler, euler_maruyama_sampler

    sampling_kwargs = dict(
        model=model,
        latents=z,
        y=y,
        num_steps=args.num_steps,
        heun=args.heun,
        cfg_scale=args.cfg_scale,
        guidance_low=args.guidance_low,
        guidance_high=args.guidance_high,
        path_type=args.path_type,
        inference_budget=inference_budget,
    )
    if args.mode == "sde":
        samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
    elif args.mode == "ode":
        samples = euler_sampler(**sampling_kwargs).to(torch.float32)
    else:
        raise NotImplementedError(f"Unknown mode: {args.mode}")
    return samples


@torch.no_grad()
def decode_latent(vae, samples, device):
    """Decode latent samples to pixel space."""
    latents_scale = torch.tensor([0.18215] * 4).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor([0.0] * 4).view(1, 4, 1, 1).to(device)
    images = vae.decode((samples - latents_bias) / latents_scale).sample
    images = (images + 1) / 2.0
    images = torch.clamp(255.0 * images, 0, 255)
    images = images.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    return images


@torch.no_grad()
def measure_flops_single_forward(model, x_dummy, t_dummy, y_dummy, inference_budget):
    """Measure FLOPs for a single forward pass of the model (not full sampling loop)."""
    with FlopCounterMode(display=False) as counter:
        model(x_dummy, t_dummy, y=y_dummy, inference_budget=inference_budget)
    return counter.get_total_flops()


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    assert torch.cuda.is_available(), "This script requires a GPU."

    # Ensure distributed is initialized (needed by ELIT masking strategy)
    _ensure_distributed()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    latent_size = args.resolution // 8

    # Build model and load checkpoint
    print(f"Building model: {args.model}")
    model = build_model(args, device)
    model = load_checkpoint(model, args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Load VAE
    print(f"Loading VAE (sd-vae-ft-{args.vae})...")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Prepare inputs (batch size 1)
    y = torch.tensor([args.class_label], device=device)

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    budgets = sorted(args.budgets)
    print(f"\nBudgets to evaluate: {budgets}")
    print(f"Warmup iterations: {args.warmup_iters}, repeat: {args.repeat}")

    # ── Warmup ──
    print("\nWarming up model...")
    for _ in range(args.warmup_iters):
        z_warmup = torch.randn(1, model.in_channels, latent_size, latent_size, device=device)
        _ = sample_single(model, z_warmup, y, args, device, inference_budget=1.0)
    torch.cuda.synchronize()
    print("Warmup complete.\n")

    # Use a fixed noise seed for all budgets so images are comparable
    rng_state = torch.cuda.get_rng_state()

    results = []

    for budget in budgets:
        budget_val = round(budget, 4)
        print(f"── Budget {budget_val:.2f} ──")

        # ── Measure FLOPs (single model forward, not full sampling loop) ──
        x_dummy = torch.randn(1, model.in_channels, latent_size, latent_size, device=device)
        t_dummy = torch.tensor([0.5], device=device, dtype=torch.float32)
        # For CFG, we double the batch
        if args.cfg_scale > 1.0:
            x_flop = torch.cat([x_dummy] * 2, dim=0)
            y_flop = torch.cat([y, torch.tensor([1000], device=device)], dim=0)
        else:
            x_flop = x_dummy
            y_flop = y
        flops = measure_flops_single_forward(model, x_flop, t_dummy.expand(x_flop.shape[0]), y_flop, budget_val)
        gflops = flops / 1e9

        # ── Measure latency (full sampling loop) ──
        latencies = []
        for r in range(args.repeat):
            torch.cuda.set_rng_state(rng_state)
            z = torch.randn(1, model.in_channels, latent_size, latent_size, device=device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            samples = sample_single(model, z, y, args, device, inference_budget=budget_val)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)

        # ── Decode and save image ──
        torch.cuda.set_rng_state(rng_state)
        z = torch.randn(1, model.in_channels, latent_size, latent_size, device=device)
        samples = sample_single(model, z, y, args, device, inference_budget=budget_val)
        pixel_images = decode_latent(vae, samples, device)

        img_path = os.path.join(images_dir, f"budget_{budget_val:.2f}.png")
        Image.fromarray(pixel_images[0]).save(img_path)

        entry = {
            "budget": budget_val,
            "gflops_per_forward": round(gflops, 2),
            "flops_per_forward": flops,
            "latency_mean_s": round(mean_latency, 4),
            "latency_std_s": round(std_latency, 4),
            "image_path": img_path,
        }
        results.append(entry)
        print(f"  GFLOPs/fwd: {gflops:.2f}  |  Latency: {mean_latency:.3f}s ± {std_latency:.3f}s  |  Saved: {img_path}")

    # ── Save results JSON ──
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ── Create figures ──
    budgets_arr = np.array([r["budget"] for r in results])
    gflops_arr = np.array([r["gflops_per_forward"] for r in results])
    latency_arr = np.array([r["latency_mean_s"] for r in results])
    latency_std_arr = np.array([r["latency_std_s"] for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Budget vs GFLOPs ──
    ax = axes[0]
    ax.plot(budgets_arr, gflops_arr, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax.set_xlabel("Inference Budget (fraction of latent tokens)", fontsize=12)
    ax.set_ylabel("GFLOPs per forward pass", fontsize=12)
    ax.set_title("Budget vs. FLOPs", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    for i, (b, g) in enumerate(zip(budgets_arr, gflops_arr)):
        ax.annotate(f"{g:.0f}", (b, g), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)

    # ── Budget vs Latency ──
    ax = axes[1]
    ax.errorbar(budgets_arr, latency_arr, yerr=latency_std_arr,
                fmt="o-", color="#FF5722", linewidth=2, markersize=8,
                capsize=4, capthick=1.5)
    ax.set_xlabel("Inference Budget (fraction of latent tokens)", fontsize=12)
    ax.set_ylabel("Latency (seconds)", fontsize=12)
    ax.set_title("Budget vs. Latency", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    for i, (b, l) in enumerate(zip(budgets_arr, latency_arr)):
        ax.annotate(f"{l:.2f}s", (b, l), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)

    plt.tight_layout()
    plt.close()

    # ── Also create individual figures ──
    # FLOPs figure
    fig_flops, ax_flops = plt.subplots(figsize=(7, 5))
    ax_flops.plot(budgets_arr, gflops_arr, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax_flops.set_xlabel("Inference Budget (fraction of latent tokens)", fontsize=12)
    ax_flops.set_ylabel("GFLOPs per forward pass", fontsize=12)
    ax_flops.set_title("Budget vs. FLOPs", fontsize=14, fontweight="bold")
    ax_flops.grid(True, alpha=0.3)
    ax_flops.set_xlim(0, 1.05)
    for b, g in zip(budgets_arr, gflops_arr):
        ax_flops.annotate(f"{g:.0f}", (b, g), textcoords="offset points",
                          xytext=(0, 10), ha="center", fontsize=8)
    fig_flops_path = os.path.join(args.output_dir, "budget_vs_flops.png")
    fig_flops.savefig(fig_flops_path, dpi=150, bbox_inches="tight")
    plt.close(fig_flops)
    print(f"FLOPs figure saved to {fig_flops_path}")

    # Latency figure
    fig_lat, ax_lat = plt.subplots(figsize=(7, 5))
    ax_lat.errorbar(budgets_arr, latency_arr, yerr=latency_std_arr,
                    fmt="o-", color="#FF5722", linewidth=2, markersize=8,
                    capsize=4, capthick=1.5)
    ax_lat.set_xlabel("Inference Budget (fraction of latent tokens)", fontsize=12)
    ax_lat.set_ylabel("Latency (seconds)", fontsize=12)
    ax_lat.set_title("Budget vs. Latency", fontsize=14, fontweight="bold")
    ax_lat.grid(True, alpha=0.3)
    ax_lat.set_xlim(0, 1.05)
    for b, l in zip(budgets_arr, latency_arr):
        ax_lat.annotate(f"{l:.2f}s", (b, l), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8)
    fig_lat_path = os.path.join(args.output_dir, "budget_vs_latency.png")
    fig_lat.savefig(fig_lat_path, dpi=150, bbox_inches="tight")
    plt.close(fig_lat)
    print(f"Latency figure saved to {fig_lat_path}")

    # ── Create image grid ──
    print("\nCreating image grid...")
    imgs = []
    for r in results:
        img = Image.open(r["image_path"])
        imgs.append(img)

    n_imgs = len(imgs)
    img_w, img_h = imgs[0].size
    cols = min(n_imgs, 5)
    rows = (n_imgs + cols - 1) // cols
    label_height = 30
    grid_w = cols * img_w
    grid_h = rows * (img_h + label_height)
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()
    except ImportError:
        draw = None
        font = None

    for idx, (img, r) in enumerate(zip(imgs, results)):
        row = idx // cols
        col = idx % cols
        x_off = col * img_w
        y_off = row * (img_h + label_height)
        grid.paste(img, (x_off, y_off))
        if draw is not None:
            label = f"b={r['budget']:.1f} | {r['gflops_per_forward']:.0f} GF | {r['latency_mean_s']:.2f}s"
            draw.text((x_off + 5, y_off + img_h + 5), label, fill=(0, 0, 0), font=font)

    grid_path = os.path.join(args.output_dir, "image_grid.png")
    grid.save(grid_path)
    print(f"Image grid saved to {grid_path}")

    print("\nDone!")

    # Cleanup distributed
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
