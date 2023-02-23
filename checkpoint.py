import os
import json
from pathlib import Path

import torch

from typing import Optional, Dict, Generator, Tuple
from collections import OrderedDict


class CheckpointManager:
    def __init__(self, path: str) -> None:
        self.path = Path(path)

        # Initial version of the checkpoints
        self.version = self.load_version()
        self.old_version = 0

        # Prepare checkpoints paths
        self.prepare()

        # Initialize training stats
        self.stats = {}

        # Initialize number of models to save
        self.n_models = 1

    def prepare(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)

    def get_version_file(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = self.path
        return path / "version.txt"

    def get_config_file(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = self.path
        return path / "config.json"

    def get_model_file(self, version: int, path: Optional[Path] = None) -> Path or Tuple:
        if path is None:
            path = self.path

        paths = []
        for n in range(self.n_models):
            paths.append(path / f"model_m{n}_v{version}.pt")
        return paths
        # return path / f"model_v{version}.pt"

    def get_stats_file(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = self.path
        return path / "training_stats.json"

    def save_config(self, config: Dict) -> None:
        config_json = json.dumps(config, indent=4)
        with self.get_config_file().open("wt") as tf:
            tf.write(config_json)

    def load_config(self) -> str:
        try:
            with self.get_config_file().open("rt") as tf:
                return tf.read()
        except FileNotFoundError as err:
            raise err

    def save_model(
            self,
            state_dicts: Dict[str, torch.Tensor] or list,
    ) -> None:
        """
        Load the model state dicts
        :param state_dicts: State dict of the model
        :return:
        """
        paths = self.get_model_file(self.version)

        if self.n_models == 1:
            torch.save(state_dicts, paths[0])
            return

        for path, dic in zip(paths, state_dicts):
            try:
                torch.save(dic, path)
            except FileNotFoundError as err:
                raise err

    def load_model(self) -> Dict[str, torch.Tensor] or list:
        """
        Load the model state dicts
        :param state_dicts: State dict of the model
        :return:
        """
        paths = self.get_model_file(self.version)

        if self.n_models == 1:
            return torch.load(paths[0])

        state_dicts = []
        for path in paths:
            try:
                state_dicts.append(torch.load(path))
            except FileNotFoundError as err:
                raise err

        return state_dicts

    def save_version(self, version: int) -> None:
        with self.get_version_file().open("wt") as tf:
            tf.write(f"{version}\n")
            tf.flush()
            os.fsync(tf.fileno())

    def load_version(self) -> int:
        try:
            with self.get_version_file().open("rt") as tf:
                version_string = tf.read().strip()
        except FileNotFoundError:
            return 0
        else:
            if len(version_string) == 0:
                return 0
            else:
                return int(version_string)

    def append_stats(self, stats: Dict) -> None:
        stats_json = json.dumps(stats)
        with self.get_stats_file().open("at") as tf:
            tf.write(f"{stats_json}\n")

    def load_stats(self) -> Generator[str, None, None]:
        try:
            with self.get_stats_file().open("rt") as tf:
                for line in tf:
                    yield line
        except FileNotFoundError as err:
            raise err

    def write_new_version(self, config: OrderedDict, state_dicts: list or OrderedDict, epoch_stats: Dict) -> None:
        if self.version == 0:
            self.save_config(config)

        # Update to new version
        self.old_version = self.version
        self.version = epoch_stats["Epoch"]
        self.save_version(self.version)

        # Save training stats here
        # Format epoch stat
        for s, v in epoch_stats.items():
            if type(v) != int:
                epoch_stats[s] = round(v, 5)

        # Save training stats here
        self.append_stats(epoch_stats)

        # Save model state dicts
        self.save_model(state_dicts)

    def remove_old_version(self) -> None:
        old_version = self.old_version

        # Remove older model
        paths = self.get_model_file(old_version)

        for path in paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass