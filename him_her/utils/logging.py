"""Persistent per-episode metric logging utilities."""

import csv
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


class EpisodeMetricsLogger:
    """Write per-episode metrics to CSV and optionally WandB."""

    def __init__(self, agent_type: str, seed: int, config: Any):
        self.agent_type = agent_type
        self.seed = seed
        self.config = config
        self.log_path = Path('logs') / f'{agent_type}_seed{seed}.csv'
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.log_path.open('w', newline='', encoding='ascii')
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames: Optional[list[str]] = None
        self._wandb_run = self._init_wandb()

    def _init_wandb(self):
        if wandb is None:
            return None

        logging_config = getattr(self.config, 'logging', None)
        project = getattr(logging_config, 'wandb_project', None)
        if not project:
            project = os.environ.get('WANDB_PROJECT')
        if not project:
            return None

        if wandb.run is not None:
            return wandb.run

        init_kwargs = {
            'project': project,
            'name': f'{self.agent_type}_seed{self.seed}',
            'reinit': True,
        }
        mode = getattr(logging_config, 'wandb_mode', None) if logging_config else None
        if mode:
            init_kwargs['mode'] = mode
        return wandb.init(**init_kwargs)

    def log(self, metrics: Dict[str, Any]) -> None:
        if self._writer is None:
            self._fieldnames = list(metrics.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()

        row = {field: metrics.get(field, '') for field in self._fieldnames}
        self._writer.writerow(row)
        self._file.flush()

        if self._wandb_run is not None:
            self._wandb_run.log(metrics)

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
        if self._wandb_run is not None and self._wandb_run is not wandb.run:
            self._wandb_run.finish()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        self.close()