from __future__ import annotations

import argparse
from typing import Tuple

import torch
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
    parser.add_argument("--sigma-policy", type=float, default=0.1, help="Gaussian policy sigma for single-policy cases")
    parser.add_argument("--sigma-exc", type=float, default=0.1, help="Sigma for excitatory policy (scenario 1.2)")
    parser.add_argument("--sigma-inh", type=float, default=0.1, help="Sigma for inhibitory policy (scenario 1.2)")
    parser.add_argument("--rho-target", type=float, default=0.1, help="Target firing rate for sparsity reward")
    parser.add_argument("--alpha-sparse", type=float, default=1.0, help="Weight for sparsity reward")
    parser.add_argument("--alpha-div", type=float, default=1.0, help="Weight for diversity reward")
    parser.add_argument("--alpha-stab", type=float, default=1.0, help="Weight for stability reward")
    parser.add_argument("--beta-margin", type=float, default=0.1, help="Margin scaling for semi-supervised reward")
    parser.add_argument("--steps", type=int, default=4, help="Number of simulation steps T for a demo run")
    parser.add_argument("--device", type=str, default="cpu", help="torch device identifier")
    return parser.parse_args()


def load_mnist_sample() -> Tuple[torch.Tensor, int]:
    """Download MNIST and return a single normalized image and label."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    image, label = dataset[0]
    return image.view(-1), int(label)


def run_demo(args: argparse.Namespace) -> None:
    """Run a lightweight demonstration of the chosen scenario on a single sample."""
    device = args.device
    image, label = load_mnist_sample()
    encoded = poisson_encode(image, args.steps)

    if args.scenario == "1.1":
        scenario = UnsupervisedSinglePolicy(
            history_length=args.history_length,
            sigma_policy=args.sigma_policy,
            rho_target=args.rho_target,
            alpha_sparse=args.alpha_sparse,
            alpha_div=args.alpha_div,
            alpha_stab=args.alpha_stab,
            num_exc_neurons=1,
            device=device,
        )
        reward = scenario.run_episode({"pre": encoded[:, 0], "post": encoded[:, 0]})
    elif args.scenario == "1.2":
        scenario = UnsupervisedDualPolicy(
            history_length=args.history_length,
            sigma_exc=args.sigma_exc,
            sigma_inh=args.sigma_inh,
            rho_target=args.rho_target,
            alpha_sparse=args.alpha_sparse,
            alpha_div=args.alpha_div,
            alpha_stab=args.alpha_stab,
            num_exc_neurons=1,
            device=device,
        )
        reward = scenario.run_episode({"pre": encoded[:, 0], "post": encoded[:, 0], "inhibitory": False})
    elif args.scenario == "2":
        scenario = SemiSupervisedScenario(
            history_length=args.history_length,
            sigma_policy=args.sigma_policy,
            beta_margin=args.beta_margin,
            device=device,
        )
        reward = scenario.run_episode({"spikes": encoded.view(args.steps, -1), "label": label})
    else:
        scenario = GradientMimicryScenario(history_length=args.history_length, sigma_policy=args.sigma_policy, device=device)
        teacher_delta = torch.zeros(1)
        agent_delta = torch.zeros(1)
        reward = scenario.run_episode(
            {
                "spikes": encoded.view(args.steps, -1),
                "layer_pos": 0.5,
                "teacher_delta": teacher_delta,
                "agent_delta": agent_delta,
            }
        )
    print(f"Demo reward for scenario {args.scenario}: {float(reward.item()):.4f}")


if __name__ == "__main__":
    args = parse_args()
    run_demo(args)
