import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import NamedTuple, Literal
import numpy as np

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
    Mesmo contrato do seu MultiVenueSOREnv, mas lendo arrays NumPy (rápido).
    Espera dicionários com arrays por coluna.
    """
    def __init__(self, lob_b3: dict, lob_base: dict, total_inventory: float = 10000, max_slippage_pct: float = 0.001):
        super().__init__()
        self.b3 = lob_b3
        self.base = lob_base
        self.total_inventory = float(total_inventory)
        self.max_slippage_pct = float(max_slippage_pct)

        self.n_steps = len(self.b3["ask_1"])
        assert self.n_steps == len(self.base["ask_1"])

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)

        self.current_step = 0
        self.inventory_remaining = self.total_inventory
        self.arrival_price = 0.0

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.inventory_remaining = self.total_inventory
        self.arrival_price = float(min(self.b3["ask_1"][0], self.base["ask_1"][0]))
        return self._get_obs(), {"inventory_left": float(self.inventory_remaining)}

    def _get_obs(self) -> np.ndarray:
        i = self.current_step
        return np.array([
            self.b3["ask_1"][i],
            self.b3["vol_ask_1"][i],
            self.base["ask_1"][i],
            self.base["vol_ask_1"][i],
            self.inventory_remaining,
        ], dtype=np.float32)

    def _execute_order(self, volume: float, venue: dict, i: int) -> ExecutionResult:
        cost, remaining = 0.0, float(volume)
        for level in range(1, 6):
            if remaining <= 0:
                break
            vol_lvl = float(venue[f"vol_ask_{level}"][i])
            px_lvl  = float(venue[f"ask_{level}"][i])
            exec_vol = min(remaining, vol_lvl)
            cost += exec_vol * px_lvl
            remaining -= exec_vol
        return ExecutionResult(cost=cost, volume_executed=float(volume) - remaining)

    def _validate_order(self, execution: ExecutionResult) -> OrderValidation:
        if execution.volume_executed <= 0:
            return OrderValidation(True, self.arrival_price, 0.0, "")

        avg_price = execution.cost / execution.volume_executed
        slippage = avg_price - self.arrival_price
        limit_px = self.arrival_price * (1 + self.max_slippage_pct)

        ok = avg_price <= limit_px
        reason = "" if ok else f"Slippage {slippage:.6f} exceeds limit {self.max_slippage_pct*100:.2f}%"
        return OrderValidation(ok, float(avg_price), float(slippage), reason)

    def _calculate_reward(self, validation: OrderValidation, inventory_executed: float, is_terminal: bool) -> float:
        if not validation.is_valid:
            return -0.5
        if inventory_executed > 0:
            impl_shortfall = -validation.slippage * inventory_executed
            opportunity_cost = -self.inventory_remaining * (self.arrival_price * 0.05) if (is_terminal and self.inventory_remaining > 0) else 0.0
            return float(impl_shortfall + opportunity_cost)
        return 0.0

    def step(self, action: Literal[0, 1, 2, 3]):
        i = self.current_step
        inv = float(self.inventory_remaining)

        if action == 0:
            execution = ExecutionResult(0.0, 0.0)
        elif action == 1:
            vol = min(100.0, inv)
            execution = self._execute_order(vol, self.b3, i) if vol > 0 else ExecutionResult(0.0, 0.0)
        elif action == 2:
            vol = min(100.0, inv)
            execution = self._execute_order(vol, self.base, i) if vol > 0 else ExecutionResult(0.0, 0.0)
        elif action == 3:
            vol_total = min(200.0, inv)
            vol_b3 = min(100.0, vol_total)
            vol_base = max(0.0, vol_total - vol_b3)
            ex_b3 = self._execute_order(vol_b3, self.b3, i) if vol_b3 > 0 else ExecutionResult(0.0, 0.0)
            ex_base = self._execute_order(vol_base, self.base, i) if vol_base > 0 else ExecutionResult(0.0, 0.0)
            execution = ExecutionResult(ex_b3.cost + ex_base.cost, ex_b3.volume_executed + ex_base.volume_executed)
        else:
            raise ValueError("Invalid action")

        validation = self._validate_order(execution)
        inventory_executed = execution.volume_executed if validation.is_valid else 0.0
        self.inventory_remaining -= inventory_executed

        self.current_step += 1
        terminated = self.inventory_remaining <= 0
        time_limit = self.current_step >= self.n_steps - 1
        truncated = bool(time_limit and not terminated)

        reward = self._calculate_reward(validation, inventory_executed, is_terminal=(terminated or truncated))

        info = {
            "inventory_left": float(self.inventory_remaining),
            "arrival_price": float(self.arrival_price),
            "executed_volume": float(inventory_executed),
            "executed_cost": float(execution.cost if validation.is_valid else 0.0),
            "avg_price": float(validation.avg_price),
            "slippage": float(validation.slippage),
            "is_valid": bool(validation.is_valid),
            "rejection_reason": validation.rejection_reason,
            "t": int(self.current_step),
            "T": int(self.n_steps - 1),
        }
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info