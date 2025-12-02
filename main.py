from typing import Callable, Dict

from scenarios.grad_mimicry import run_grad
from scenarios.semi_supervised import run_semi
from scenarios.unsup_dual import run_unsup2
from scenarios.unsup_single import run_unsup1
from utils.cli import parse_args
from utils.logging import create_result_dir, get_logger
from utils.seeds import set_global_seed


def main():
    args = parse_args()
    set_global_seed(args.seed)

    result_dir = create_result_dir(args.scenario)
    args.result_dir = result_dir
    logger = get_logger(result_dir)

    scenario_runners: Dict[str, Callable] = {
        "unsup1": run_unsup1,
        "unsup2": run_unsup2,
        "semi": run_semi,
        "grad": run_grad,
    }

    logger.info("Starting scenario: %s", args.scenario)
    logger.info("Results will be saved to: %s", result_dir)

    runner = scenario_runners.get(args.scenario)
    if runner is None:
        raise ValueError(f"Unknown scenario: {args.scenario}")

    runner(args, logger)


if __name__ == "__main__":
    main()
