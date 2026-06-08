from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, NamedTuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ExecutionResult(NamedTuple):
    cost: float
    volume_executed: float


@dataclass(frozen=True)
class OrderValidation:
    is_valid: bool
    avg_price: float
    slippage: float
    rejection_reason: str = ""


class MultiVenueSOREnvNumpy(gym.Env):
    """
    Ambiente SOR com L2 em memória (NumPy), compatível com Gymnasium.

    Estado (5D):
        [ask_1_b3, vol_ask_1_b3, ask_1_base, vol_ask_1_base, inventory_remaining]

    Ações:
        0 = Wait
        1 = Executa até 100 no B3
        2 = Executa até 100 no BASE
        3 = Slice: até 200 (100 B3 + 100 BASE)

    Observações:
        - Este ambiente NÃO lê parquet; recebe dicionários de arrays prontos.
        - Para dados reais particionados, use o ambiente parquet/factory do projeto.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        lob_b3: Mapping[str, np.ndarray],
        lob_base: Mapping[str, np.ndarray],
        total_inventory: float = 10_000,
        max_slippage_pct: float = 0.001,
    ) -> None:
        super().__init__()

        self._required_keys = tuple(
            [f"ask_{i}" for i in range(1, 6)] + [f"vol_ask_{i}" for i in range(1, 6)]
        )

        self.b3 = self._normalize_l2(lob_b3, name="lob_b3")
        self.base = self._normalize_l2(lob_base, name="lob_base")

        self.total_inventory = float(total_inventory)
        self.max_slippage_pct = float(max_slippage_pct)

        self.n_steps = len(self.b3["ask_1"])
        if self.n_steps != len(self.base["ask_1"]):
            raise ValueError(
                f"lob_b3 e lob_base têm tamanhos diferentes: {self.n_steps} vs {len(self.base['ask_1'])}"
            )
        if self.n_steps < 1:
            raise ValueError("Séries L2 vazias: é necessário ao menos 1 step.")

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(5,), dtype=np.float32
        )

        self.current_step = 0
        self.inventory_remaining = self.total_inventory
        self.arrival_price = 0.0
        self._done = False

    # ---------------------------------------------------------------------
    # API Gymnasium
    # ---------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.current_step = 0
        self.inventory_remaining = float(self.total_inventory)
        self.arrival_price = float(min(self.b3["ask_1"][0], self.base["ask_1"][0]))
        self._done = False

        obs = self._get_obs(self.current_step)
        info = {"inventory_left": float(self.inventory_remaining)}
        return obs, info

    def step(self, action: Literal[0, 1, 2, 3]):
        if self._done:
            raise RuntimeError(
                "step() chamado após término do episódio. Execute reset() antes de novo episódio."
            )

        if not self.action_space.contains(int(action)):
            raise ValueError(f"Ação inválida: {action}")

        i = self.current_step
        inv = float(self.inventory_remaining)

        # -------- execução da ação --------
        if action == 0:  # wait
            execution = ExecutionResult(0.0, 0.0)

        elif action == 1:  # B3
            vol = min(100.0, inv)
            execution = self._execute_order(vol, self.b3, i) if vol > 0 else ExecutionResult(0.0, 0.0)

        elif action == 2:  # BASE
            vol = min(100.0, inv)
            execution = self._execute_order(vol, self.base, i) if vol > 0 else ExecutionResult(0.0, 0.0)

        else:  # action == 3 -> slice
            vol_total = min(200.0, inv)
            vol_b3 = min(100.0, vol_total)
            vol_base = max(0.0, vol_total - vol_b3)

            ex_b3 = self._execute_order(vol_b3, self.b3, i) if vol_b3 > 0 else ExecutionResult(0.0, 0.0)
            ex_base = self._execute_order(vol_base, self.base, i) if vol_base > 0 else ExecutionResult(0.0, 0.0)

            execution = ExecutionResult(
                cost=ex_b3.cost + ex_base.cost,
                volume_executed=ex_b3.volume_executed + ex_base.volume_executed,
            )

        # -------- validação / inventário --------
        validation = self._validate_order(execution)
        inventory_executed = execution.volume_executed if validation.is_valid else 0.0
        self.inventory_remaining = max(0.0, self.inventory_remaining - inventory_executed)

        # Avança tempo (após executar no índice i)
        self.current_step += 1

        terminated = bool(self.inventory_remaining <= 0.0)
        time_limit = bool(self.current_step >= self.n_steps)  # corrigido: usa todo o array
        truncated = bool(time_limit and not terminated)

        reward = self._calculate_reward(
            validation=validation,
            inventory_executed=inventory_executed,
            is_terminal=(terminated or truncated),
        )

        self._done = bool(terminated or truncated)

        info = {
            "inventory_left": float(self.inventory_remaining),
            "arrival_price": float(self.arrival_price),
            "executed_volume": float(inventory_executed),
            "executed_cost": float(execution.cost if validation.is_valid else 0.0),
            "avg_price": float(validation.avg_price),
            "slippage": float(validation.slippage),
            "is_valid": bool(validation.is_valid),
            "rejection_reason": validation.rejection_reason,
            "t": int(min(self.current_step, self.n_steps)),
            "T": int(self.n_steps),
        }

        # Evita index out of bounds no estado terminal
        if self._done:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            next_obs = self._get_obs(self.current_step)

        return next_obs, float(reward), terminated, truncated, info

    # ---------------------------------------------------------------------
    # Internos
    # ---------------------------------------------------------------------
    def _normalize_l2(self, lob: Mapping[str, np.ndarray], name: str) -> dict[str, np.ndarray]:
        missing = [k for k in self._required_keys if k not in lob]
        if missing:
            raise KeyError(f"{name} sem colunas obrigatórias: {missing}")

        out: dict[str, np.ndarray] = {}
        lengths = set()

        for k in self._required_keys:
            arr = np.asarray(lob[k], dtype=np.float32)
            if arr.ndim != 1:
                raise ValueError(f"{name}[{k}] deve ser vetor 1D, recebido shape={arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name}[{k}] contém NaN/Inf")
            out[k] = arr
            lengths.add(len(arr))

        if len(lengths) != 1:
            raise ValueError(f"{name} possui colunas com tamanhos diferentes: {sorted(lengths)}")

        if out["ask_1"][0] <= 0 or out["ask_1"].min() <= 0:
            raise ValueError(f"{name} possui preços ask não positivos em ask_1")

        return out

    def _get_obs(self, i: int) -> np.ndarray:
        return np.array(
            [
                self.b3["ask_1"][i],
                self.b3["vol_ask_1"][i],
                self.base["ask_1"][i],
                self.base["vol_ask_1"][i],
                self.inventory_remaining,
            ],
            dtype=np.float32,
        )

    def _execute_order(
        self,
        volume: float,
        venue: Mapping[str, np.ndarray],
        i: int,
    ) -> ExecutionResult:
        cost = 0.0
        remaining = float(volume)

        for level in range(1, 6):
            if remaining <= 0:
                break

            vol_lvl = float(venue[f"vol_ask_{level}"][i])
            px_lvl = float(venue[f"ask_{level}"][i])

            if vol_lvl <= 0:
                continue

            exec_vol = min(remaining, vol_lvl)
            cost += exec_vol * px_lvl
            remaining -= exec_vol

        return ExecutionResult(cost=float(cost), volume_executed=float(volume - remaining))

    def _validate_order(self, execution: ExecutionResult) -> OrderValidation:
        if execution.volume_executed <= 0:
            return OrderValidation(
                is_valid=True,
                avg_price=float(self.arrival_price),
                slippage=0.0,
                rejection_reason="",
            )

        avg_price = float(execution.cost / execution.volume_executed)
        slippage = float(avg_price - self.arrival_price)
        limit_px = float(self.arrival_price * (1.0 + self.max_slippage_pct))

        ok = bool(avg_price <= limit_px)
        reason = "" if ok else (
            f"Slippage {slippage:.6f} excede limite de {self.max_slippage_pct*100:.3f}%"
        )
        return OrderValidation(ok, avg_price, slippage, reason)

    def _calculate_reward(
        self,
        validation: OrderValidation,
        inventory_executed: float,
        is_terminal: bool,
    ) -> float:
        # Penalidade por ordem inválida/rejeitada
        if not validation.is_valid:
            return -0.5

        # Sem execução, sem custo imediato
        if inventory_executed <= 0:
            if is_terminal and self.inventory_remaining > 0:
                # penalidade terminal por não execução total
                return float(-self.inventory_remaining * (self.arrival_price * 0.05))
            return 0.0

        # Implementation shortfall (quanto menor slippage, melhor)
        impl_shortfall = -validation.slippage * inventory_executed

        # Penalidade terminal por sobrar inventário
        opportunity_cost = (
            -self.inventory_remaining * (self.arrival_price * 0.05)
            if (is_terminal and self.inventory_remaining > 0)
            else 0.0
        )

        return float(impl_shortfall + opportunity_cost)