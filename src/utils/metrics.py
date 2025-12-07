import math
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from utils.logging import resolve_path


def _prepare_output_path(path: str | Path) -> Path:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _to_numpy(array: torch.Tensor | np.ndarray | Iterable) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def plot_delta_t_delta_d(delta_t: torch.Tensor | np.ndarray, delta_d: torch.Tensor | np.ndarray, out_path: str) -> None:
    out_path = _prepare_output_path(out_path)
    dt_np = _to_numpy(delta_t).flatten()
    dd_np = _to_numpy(delta_d).flatten()
    base = out_path.with_suffix("")

    plt.figure(figsize=(6, 5))
    plt.scatter(dt_np, dd_np, alpha=0.6, s=10)
    plt.xlabel(r"$\Delta t$")
    plt.ylabel(r"$\Delta d$")
    plt.title("Spike timing vs weight change")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    with (base.with_name(base.name + "_metrics.txt")).open("w") as f:
        corr = np.corrcoef(dt_np, dd_np)[0, 1] if dt_np.size > 1 else float("nan")
        f.write(f"count\t{dt_np.size}\n")
        f.write(f"corr\t{corr:.6f}\n")


def plot_weight_histograms(
    weights_before: torch.Tensor | np.ndarray, weights_after: torch.Tensor | np.ndarray, out_path: str, bins: int = 50
) -> None:
    out_path = _prepare_output_path(out_path)
    w_before = _to_numpy(weights_before).flatten()
    w_after = _to_numpy(weights_after).flatten()
    base = out_path.with_suffix("")

    plt.figure(figsize=(7, 5))
    plt.hist(w_before, bins=bins, alpha=0.6, label="before", density=True)
    plt.hist(w_after, bins=bins, alpha=0.6, label="after", density=True)
    plt.xlabel("Weight value")
    plt.ylabel("Density")
    plt.title("Weight distribution before/after training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    np.savez(base.with_suffix(".npz"), weights_before=w_before, weights_after=w_after)
    with (base.with_name(base.name + "_stats.txt")).open("w") as f:
        f.write(
            "before_mean\t{:.6f}\nbefore_std\t{:.6f}\nafter_mean\t{:.6f}\nafter_std\t{:.6f}\n".format(
                float(np.mean(w_before)), float(np.std(w_before)), float(np.mean(w_after)), float(np.std(w_after))
            )
        )


def plot_receptive_fields(weights: torch.Tensor, out_path: str) -> None:
    out_path = _prepare_output_path(out_path)
    n_neurons = weights.shape[1]
    nrow = max(1, math.isqrt(n_neurons))

    weight_imgs = weights.t().reshape(n_neurons, 1, 28, 28)
    grid = torchvision.utils.make_grid(weight_imgs, nrow=nrow, normalize=True)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="hot")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def compute_neuron_labels(firing_rates: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    if firing_rates.dim() != 2:
        raise ValueError("firing_rates must be 2D (batch, neurons)")
    label_responses = torch.zeros((firing_rates.size(1), num_classes), device=firing_rates.device)
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            label_responses[:, c] = firing_rates[mask].sum(dim=0)
    neuron_labels = label_responses.argmax(dim=1)
    return neuron_labels


def evaluate_labeling(
    firing_rates: torch.Tensor, labels: torch.Tensor, neuron_labels: torch.Tensor, num_classes: int, out_prefix: str
) -> Tuple[float, np.ndarray]:
    out_prefix = _prepare_output_path(out_prefix)
    with torch.no_grad():
        winners = firing_rates.argmax(dim=1)
        preds = neuron_labels[winners]
        accuracy = (preds == labels).float().mean().item()

    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    conf = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels_np, preds_np):
        conf[t, p] += 1

    np.savez(out_prefix.with_suffix(".npz"), confusion=conf, preds=preds_np, labels=labels_np)
    with (out_prefix.with_name(out_prefix.name + "_metrics.txt")).open("w") as f:
        f.write(f"accuracy\t{accuracy:.6f}\n")

    plt.figure(figsize=(6, 5))
    plt.imshow(conf, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".png"))
    plt.close()

    return accuracy, conf


def plot_grad_alignment(
    delta_agent: torch.Tensor | np.ndarray, delta_teacher: torch.Tensor | np.ndarray, out_path: str
) -> None:
    out_path = _prepare_output_path(out_path)
    d_agent = _to_numpy(delta_agent).flatten()
    d_teacher = _to_numpy(delta_teacher).flatten()
    base = out_path.with_suffix("")

    plt.figure(figsize=(6, 5))
    plt.scatter(d_agent, d_teacher, alpha=0.6, s=8)
    plt.xlabel(r"$\Delta w_{agent}$")
    plt.ylabel(r"$\Delta w_{teacher}$")
    plt.title("Gradient alignment")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    np.savez(base.with_suffix(".npz"), delta_agent=d_agent, delta_teacher=d_teacher)

    mse = np.mean((d_agent - d_teacher) ** 2)
    with (base.with_name(base.name + "_metrics.txt")).open("w") as f:
        f.write(f"mse\t{mse:.6f}\n")
