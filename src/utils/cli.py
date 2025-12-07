import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SNN PPO experiments on MNIST")
    parser.add_argument(
        "--scenario",
        choices=["unsup1", "unsup2", "semi", "grad"],
        required=True,
        help="Select experiment scenario",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--batch-size-images",
        type=int,
        default=16,
        help="Mini-batch size in number of images",
    )
    parser.add_argument(
        "--event-batch-size",
        type=int,
        default=1024,
        help="Mini-batch size for processing flattened events with the actor/critic",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Name for the current run / result directory")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Logging interval in epochs/episodes for console/file logs",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of epochs for training (default for smoke tests)",
    )
    parser.add_argument("--T-unsup1", type=int, default=100, help="Timesteps for unsupervised scenario 1")
    parser.add_argument("--T-unsup2", type=int, default=100, help="Timesteps for unsupervised scenario 2")
    parser.add_argument("--T-semi", type=int, default=100, help="Timesteps for semi-supervised scenario")
    parser.add_argument("--T-sup", type=int, default=100, help="Timesteps for supervised gradient mimicry")
    parser.add_argument("--spike-array-len", type=int, default=20, help="Spike history length L for policy inputs")
    parser.add_argument("--sigma-unsup1", type=float, default=0.1, help="Gaussian sigma for unsup1 policy")
    parser.add_argument("--sigma-unsup2", type=float, default=0.1, help="Gaussian sigma for unsup2 policies")
    parser.add_argument("--sigma-semi", type=float, default=0.1, help="Gaussian sigma for semi-supervised policy")
    parser.add_argument("--sigma-sup", type=float, default=0.1, help="Gaussian sigma for gradient mimicry policy")
    parser.add_argument("--N-E", type=int, default=100, help="Number of excitatory neurons in Diehl–Cook network")
    parser.add_argument("--N-hidden", type=int, default=256, help="Hidden layer size for semi-supervised network")
    parser.add_argument("--dt", type=float, default=1.0, help="Simulation time step Δt for LIF dynamics")
    parser.add_argument(
        "--layer-index-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to normalized layer indices in grad mimicry scenario",
    )
    parser.add_argument("--lr-actor", type=float, default=1e-3, help="Learning rate for actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="Learning rate for critic")
    parser.add_argument("--ppo-eps", type=float, default=0.2, help="PPO clipping epsilon")
    parser.add_argument("--ppo-epochs", type=int, default=2, help="PPO epochs per update")
    parser.add_argument(
        "--ppo-batch-size",
        type=int,
        default=512,
        help="Mini-batch size inside PPO update (larger for stable event-driven SGD)",
    )
    parser.add_argument(
        "--exc-clip-min",
        type=float,
        default=0.0,
        help="Lower bound for excitatory synapse weights (unsupervised scenarios)",
    )
    parser.add_argument(
        "--exc-clip-max",
        type=float,
        default=1.0,
        help="Upper bound for excitatory synapse weights (unsupervised scenarios)",
    )
    parser.add_argument(
        "--grad-clip-min",
        type=float,
        default=0.0,
        help="Lower bound for supervised/semi-supervised weight updates",
    )
    parser.add_argument(
        "--grad-clip-max",
        type=float,
        default=1.0,
        help="Upper bound for supervised/semi-supervised weight updates",
    )
    parser.add_argument(
        "--inh-clip-min",
        type=float,
        default=0.0,
        help="Lower bound for inhibitory synapse weights (magnitude before sign)",
    )
    parser.add_argument(
        "--inh-clip-max",
        type=float,
        default=1.0,
        help="Upper bound for inhibitory synapse weights (magnitude before sign)",
    )
    parser.add_argument(
        "--local-lr",
        type=float,
        default=0.01,
        help="Local learning rate η_w used in Δw_i(t) = η_w * s_scen * Δd_i(t)",
    )
    parser.add_argument("--rho-target", type=float, default=0.1, help="Target firing rate for sparsity reward")
    parser.add_argument("--alpha-sparse", type=float, default=1.0, help="Weight for sparsity reward")
    parser.add_argument("--alpha-div", type=float, default=0.1, help="Weight for diversity reward")
    parser.add_argument("--alpha-stab", type=float, default=0.1, help="Weight for stability reward")
    parser.add_argument("--beta-margin", type=float, default=0.5, help="Weight for margin reward")
    parser.add_argument("--alpha-align", type=float, default=0.1, help="Alignment step size for gradient mimicry")
    parser.add_argument("--max-rate", type=float, default=1.0, help="Maximal firing rate for Poisson encoder")
    parser.add_argument(
        "--direct-input",
        action="store_true",
        help="If set, feed real-valued inputs (currents) instead of Poisson spikes in grad scenario",
    )
    parser.add_argument(
        "--log-gradient-stats",
        action="store_true",
        help="Enable extended logging of gradient/Δw statistics in grad mimicry scenario",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args()
