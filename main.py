from __future__ import annotations

import argparse
import random

import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.scenarios.semi_supervised import SemiSupervisedScenario
from src.scenarios.supervised import GradientMimicryScenario
from src.scenarios.unsupervised import UnsupervisedDualPolicy, UnsupervisedSinglePolicy
from src.utils.poisson_encoding import poisson_encode


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for selecting scenarios and hyperparameters."""
    parser = argparse.ArgumentParser(description="RL-driven synaptic plasticity experiments")
    parser.add_argument("--scenario", choices=["1.1", "1.2", "2", "3"], required=True, help="Experiment scenario identifier")
    parser.add_argument("--run-name", type=str, default="default", help="Run name used for logging")
    parser.add_argument("--history-length", type=int, default=16, help="Spike history length L")
    parser.add_argument("--dt", type=float, default=1.0, help="Simulation timestep for LIF neurons")
    parser.add_argument("--N-E", type=int, default=1, dest="N_E", help="Number of excitatory neurons")
    parser.add_argument("--N-hidden", type=int, default=0, dest="N_hidden", help="Number of hidden neurons")
    parser.add_argument("--lif-tau-m", type=float, default=20.0, dest="lif_tau_m", help="LIF membrane time constant")
    parser.add_argument("--lif-v-threshold", type=float, default=1.0, dest="lif_v_threshold", help="LIF firing threshold")
    parser.add_argument("--lif-v-reset", type=float, default=0.0, dest="lif_v_reset", help="LIF reset potential")
    parser.add_argument("--lr-actor", type=float, default=1e-3, dest="lr_actor", help="Actor learning rate")
    parser.add_argument("--lr-critic", type=float, default=1e-3, dest="lr_critic", help="Critic learning rate")
    parser.add_argument("--alpha-align", type=float, default=1.0, dest="alpha_align", help="Gradient mimicry alignment scale")
    parser.add_argument("--log-gradient-stats", action="store_true", help="Enable gradient statistics logging")
    parser.add_argument("--T-unsup1", type=int, default=4, dest="T_unsup1", help="Scenario 1.1 timestep count")
    parser.add_argument("--T-unsup2", type=int, default=4, dest="T_unsup2", help="Scenario 1.2 timestep count")
    parser.add_argument("--T-semi", type=int, default=4, dest="T_semi", help="Scenario 2 timestep count")
    parser.add_argument("--T-sup", type=int, default=4, dest="T_sup", help="Scenario 3 timestep count")
    parser.add_argument("--sigma-unsup1", type=float, default=0.1, dest="sigma_unsup1", help="Gaussian sigma for scenario 1.1")
    parser.add_argument("--sigma-unsup2-exc", type=float, default=0.1, dest="sigma_unsup2_exc", help="Excitatory sigma for scenario 1.2")
    parser.add_argument("--sigma-unsup2-inh", type=float, default=0.1, dest="sigma_unsup2_inh", help="Inhibitory sigma for scenario 1.2")
    parser.add_argument("--sigma-semi", type=float, default=0.1, dest="sigma_semi", help="Gaussian sigma for scenario 2")
    parser.add_argument("--sigma-sup", type=float, default=0.1, dest="sigma_sup", help="Gaussian sigma for scenario 3")
    parser.add_argument("--rho-target", type=float, default=0.1, help="Target firing rate for sparsity reward")
    parser.add_argument("--alpha-sparse", type=float, default=1.0, help="Weight for sparsity reward")
    parser.add_argument("--alpha-div", type=float, default=1.0, help="Weight for diversity reward")
    parser.add_argument("--alpha-stab", type=float, default=1.0, help="Weight for stability reward")
    parser.add_argument("--beta-margin", type=float, default=0.1, help="Margin scaling for semi-supervised reward")
    parser.add_argument("--exc-clip-min", type=float, default=-1.0, dest="exc_clip_min", help="Excitatory clip lower bound")
    parser.add_argument("--exc-clip-max", type=float, default=1.0, dest="exc_clip_max", help="Excitatory clip upper bound")
    parser.add_argument("--inh-clip-min", type=float, default=-1.0, dest="inh_clip_min", help="Inhibitory clip lower bound")
    parser.add_argument("--inh-clip-max", type=float, default=1.0, dest="inh_clip_max", help="Inhibitory clip upper bound")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training episodes to run")
    parser.add_argument("--log-interval", type=int, default=100, dest="log_interval", help="Batches between log messages")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cpu", help="torch device identifier")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_mnist_dataloader(batch_size: int = 1) -> DataLoader:
    """Return a DataLoader that iterates over the full MNIST training set."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def run_training(args: argparse.Namespace) -> None:
    """Iterate across the MNIST dataset and execute the selected scenario."""
    set_seed(args.seed)
    device = args.device
    dataloader = build_mnist_dataloader(batch_size=1)

    reward = torch.tensor(0.0)
    if args.scenario == "1.1":
        steps = args.T_unsup1
        scenario = UnsupervisedSinglePolicy(
            history_length=args.history_length,
            sigma_policy=args.sigma_unsup1,
            rho_target=args.rho_target,
            alpha_sparse=args.alpha_sparse,
            alpha_div=args.alpha_div,
            alpha_stab=args.alpha_stab,
            num_exc_neurons=args.N_E,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            run_name=args.run_name,
            device=device,
        )
    elif args.scenario == "1.2":
        steps = args.T_unsup2
        scenario = UnsupervisedDualPolicy(
            history_length=args.history_length,
            sigma_exc=args.sigma_unsup2_exc,
            sigma_inh=args.sigma_unsup2_inh,
            rho_target=args.rho_target,
            alpha_sparse=args.alpha_sparse,
            alpha_div=args.alpha_div,
            alpha_stab=args.alpha_stab,
            num_exc_neurons=args.N_E,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            run_name=args.run_name,
            device=device,
        )
    elif args.scenario == "2":
        steps = args.T_semi
        scenario = SemiSupervisedScenario(
            history_length=args.history_length,
            sigma_policy=args.sigma_semi,
            beta_margin=args.beta_margin,
            num_hidden=args.N_hidden,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            run_name=args.run_name,
            device=device,
        )
    else:
        steps = args.T_sup
        scenario = GradientMimicryScenario(
            history_length=args.history_length,
            sigma_policy=args.sigma_sup,
            alpha_align=args.alpha_align,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            lif_tau_m=args.lif_tau_m,
            lif_v_threshold=args.lif_v_threshold,
            lif_v_reset=args.lif_v_reset,
            dt=args.dt,
            log_gradient_stats=args.log_gradient_stats,
            run_name=args.run_name,
            device=device,
        )

    for epoch in range(args.num_epochs):
        for batch_idx, (image, label) in enumerate(dataloader):
            image = image.to(device)
            label = label.to(device)

            encoded = poisson_encode(image.view(-1), steps)
            if args.scenario == "1.1":
                reward = scenario.run_episode(
                    {
                        "pre": encoded[:, 0],
                        "post": encoded[:, 0],
                        "clip": (args.exc_clip_min, args.exc_clip_max),
                    }
                )
            elif args.scenario == "1.2":
                reward = scenario.run_episode(
                    {
                        "pre_exc": encoded[:, 0],
                        "post_exc": encoded[:, 0],
                        "clip_exc": (args.exc_clip_min, args.exc_clip_max),
                        "clip_inh": (args.inh_clip_min, args.inh_clip_max),
                    }
                )
            elif args.scenario == "2":
                reward = scenario.run_episode(
                    {
                        "spikes": encoded.view(steps, -1),
                        "label": int(label.item()),
                        "clip": (args.exc_clip_min, args.exc_clip_max),
                    }
                )
            else:
                reward = scenario.run_episode(
                    {
                        "image": image.squeeze(0),
                        "steps": steps,
                        "layer_pos": 0.5,
                        "label": int(label.item()),
                        "clip": (args.exc_clip_min, args.exc_clip_max),
                    }
                )
            if batch_idx % args.log_interval == 0 or batch_idx == len(dataloader) - 1:
                print(
                    f"Epoch {epoch + 1}/{args.num_epochs} batch {batch_idx + 1}/{len(dataloader)} "
                    f"reward for scenario {args.scenario}: {float(reward.item()):.4f}"
                )


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
