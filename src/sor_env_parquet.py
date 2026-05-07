from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from src.sor_env_numpy import MultiVenueSOREnvNumpy
from src.offline_dataset import (
    list_available_dates, load_l2_day_df, df_to_l2_numpy, slice_l2_numpy
)

class MultiVenueSOREnvParquet(gym.Env):
    """
    Env que amostra episódios de um dataset Parquet particionado (12 meses).
    Em cada reset():
      - escolhe um dia aleatório
      - escolhe um start aleatório
      - carrega slice para B3 e Base
      - cria um MultiVenueSOREnvNumpy interno para rodar o episódio
    """

    def __init__(
        self,
        root: str,
        symbol: str,
        venue_b3: str = "B3",
        venue_base: str = "BASE",
        episode_len: int = 500,
        total_inventory: float = 10_000,
        max_slippage_pct: float = 0.001,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.symbol = symbol
        self.venue_b3 = venue_b3
        self.venue_base = venue_base
        self.episode_len = int(episode_len)
        self.total_inventory = float(total_inventory)
        self.max_slippage_pct = float(max_slippage_pct)

        self._rng = random.Random(seed)
        self._dates = list_available_dates(root, venue_b3, symbol)
        if not self._dates:
            raise RuntimeError("Nenhuma date= encontrada para o ativo/venue_b3.")

        self._inner: Optional[MultiVenueSOREnvNumpy] = None

        # expõe os spaces do env interno
        # (apenas placeholders até primeiro reset)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._rng.seed(seed)

        date = self._rng.choice(self._dates)

        df_b3 = load_l2_day_df(self.root, self.venue_b3, self.symbol, date)
        df_base = load_l2_day_df(self.root, self.venue_base, self.symbol, date)

        if len(df_b3) != len(df_base):
            n = min(len(df_b3), len(df_base))
            df_b3 = df_b3.iloc[:n].reset_index(drop=True)
            df_base = df_base.iloc[:n].reset_index(drop=True)

        n_steps = len(df_b3)
        if n_steps <= self.episode_len:
            start = 0
            length = n_steps
        else:
            start = self._rng.randint(0, n_steps - self.episode_len - 1)
            length = self.episode_len

        lob_b3 = slice_l2_numpy(df_to_l2_numpy(df_b3), start, length)
        lob_base = slice_l2_numpy(df_to_l2_numpy(df_base), start, length)

        self._inner = MultiVenueSOREnvNumpy(
            lob_b3=lob_b3,
            lob_base=lob_base,
            total_inventory=self.total_inventory,
            max_slippage_pct=self.max_slippage_pct,
        )

        obs, info = self._inner.reset()
        self._sync_public_attrs()
        return obs, info

    def _sync_public_attrs(self) -> None:
        # expõe atributos úteis para wrappers/avaliação
        if self._inner is None:
            return
        self.current_step = self._inner.current_step
        self.n_steps = self._inner.n_steps
        self.inventory_remaining = self._inner.inventory_remaining
        self.total_inventory = self._inner.total_inventory
        self.arrival_price = self._inner.arrival_price

    def step(self, action: int):
        if self._inner is None:
            raise RuntimeError("Env não inicializado. Chame reset() antes de step().")
        obs, reward, terminated, truncated, info = self._inner.step(action)
        self._sync_public_attrs()
        return obs, reward, terminated, truncated, info