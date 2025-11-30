from __future__ import annotations

import argparse

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
    parser.add_argument("--history-length", type=int, default=16, help="Spike history length L")
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
    parser.add_argument("--device", type=str, default="cpu", help="torch device identifier")
    return parser.parse_args()


def build_mnist_dataloader(batch_size: int = 1) -> DataLoader:
    """Return a DataLoader that iterates over the full MNIST training set."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def run_training(args: argparse.Namespace) -> None:
    """Iterate across the MNIST dataset and execute the selected scenario."""
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
            num_exc_neurons=1,
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
            num_exc_neurons=1,
            device=device,
        )
    elif args.scenario == "2":
        steps = args.T_semi
        scenario = SemiSupervisedScenario(
            history_length=args.history_length,
            sigma_policy=args.sigma_semi,
            beta_margin=args.beta_margin,
            device=device,
        )
    else:
        steps = args.T_sup
        scenario = GradientMimicryScenario(history_length=args.history_length, sigma_policy=args.sigma_sup, device=device)

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
            print(
                f"Epoch {epoch + 1}/{args.num_epochs} batch {batch_idx + 1}/{len(dataloader)} "
                f"reward for scenario {args.scenario}: {float(reward.item()):.4f}"
            )


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
