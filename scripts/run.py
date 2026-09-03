"""Hydra entry point for the Task 1/2/3 runners.

This is the composition-friendly front door to the same runners the JSON CLIs
drive. It composes `conf/`, converts the result into the canonical run-config
dictionary (`scripts/hydra_bridge.py`), and dispatches:

    task=task1|task2|both  -> scripts/run_experiment_from_config.py
    task=task3             -> scripts/run_task3_verification_from_config.py

Examples
--------
    .venv/bin/python scripts/run.py profile=zai model=glm-5.1 dataset=nice
    .venv/bin/python scripts/run.py --multirun profile=zai \
        model=glm-5.1,glm-4.7 dataset=nice,mlm_tapt mode=smoke fake_completion=true
    .venv/bin/python scripts/run.py task=task3 source_run_id=full-... mode=full

Hydra never changes the process working directory (`hydra.job.chdir: false`):
every path in this project is resolved through `eval_utils.project_root()`.
The legacy JSON path (`--config run_configs/*.json`) is unchanged and remains
the provenance record for the published runs.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

try:
    import eval_utils as eu
    import hydra_bridge as hb
    import run_experiment_from_config as experiment_runner
    import run_task3_verification_from_config as task3_runner
except ModuleNotFoundError:  # pragma: no cover
    from scripts import (
        eval_utils as eu,
        hydra_bridge as hb,
        run_experiment_from_config as experiment_runner,
        run_task3_verification_from_config as task3_runner,
    )


def dispatch(cfg: DictConfig) -> None:
    """Run one composed cell (one Hydra job)."""
    eu.configure_run_logging(str(cfg.log_level))
    run_config = eu.normalize_run_config(hb.hydra_config_to_run_config(cfg))
    args = hb.run_args_namespace(cfg, hb.resolved_config_yaml(cfg))
    tasks = eu.normalize_task_filter(str(cfg.task))
    if tasks == ["task3"]:
        if not args.source_run_id:
            raise ValueError("task=task3 requires source_run_id=<complete task2 run id>.")
        task3_runner.run_from_config(run_config, args)
        return
    experiment_runner.run_from_config(run_config, args)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    dispatch(cfg)


if __name__ == "__main__":
    main()
