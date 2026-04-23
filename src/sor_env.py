import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import NamedTuple, Literal
import numpy as np
import pandas as pd


class ExecutionResult(NamedTuple):
    """Resultado da execução de uma ordem no LOB"""
    cost: float
    volume_executed: float


@dataclass(frozen=True)
class OrderValidation:
    """Validação do resultado da ordem"""
    is_valid: bool
    avg_price: float
    slippage: float
    rejection_reason: str = ""

class MultiVenueSOREnv(gym.Env):   
    """
    Ambiente de Simulação SOR para Mercado Fragmentado (B3 + Base Exchange).
    Inclui Filtro de Slippage Máximo e Agregação de Liquidez.
    """
    def __init__(
        self, 
        lob_b3: pd.DataFrame, 
        lob_base: pd.DataFrame, 
        total_inventory: float = 10000, 
        max_slippage_pct: float = 0.001
    ) -> None:
        super(MultiVenueSOREnv, self).__init__()
        self.lob_b3 = lob_b3       # DataFrame Level 2 da B3
        self.lob_base = lob_base   # DataFrame Level 2 da Base Exchange
        self.total_inventory = total_inventory
        self.max_slippage_pct = max_slippage_pct # Ex: 0.10% de limite
        self.current_step = 0

        # Ações do SOR: 
        # 0 = Aguardar (Evitar HFT Gaming)
        # 1 = Comprar 100 na B3
        # 2 = Comprar 100 na Base Exchange
        # 3 = Comprar 200 (Slicing: 100 na B3 + 100 na Base)
        self.action_space = spaces.Discrete(4)

        # Estado (5D): [Ask_B3, Vol_B3, Ask_Base, Vol_Base, Inv_t]
        # Focamos no lado da Venda (Ask) pois o fundo está comprando.
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.inventory_remaining = self.total_inventory

        # Arrival Price (Fita Consolidada - Melhor preço global no instante zero)
        ask_b3_inicial = self.lob_b3.iloc[0]['ask_1']
        ask_base_inicial = self.lob_base.iloc[0]['ask_1']
        self.arrival_price = min(ask_b3_inicial, ask_base_inicial)

        return self._get_obs(), {'inventory_left': self.inventory_remaining}

    def _get_obs(self) -> np.ndarray:
        row_b3 = self.lob_b3.iloc[self.current_step]
        row_base = self.lob_base.iloc[self.current_step]
        return np.array([
            row_b3['ask_1'], 
            row_b3['vol_ask_1'], 
            row_base['ask_1'], 
            row_base['vol_ask_1'], 
            self.inventory_remaining
        ], dtype=np.float32)

    def _execute_order(self, volume: float, row_data: pd.Series) -> ExecutionResult:
        """Simula o consumo do LOB (Level 2) em uma bolsa específica"""
        cost, remaining = 0.0, volume
        for level in range(1, 6):
            if remaining <= 0:
                break
            exec_vol = min(remaining, row_data[f'vol_ask_{level}'])
            cost += exec_vol * row_data[f'ask_{level}']
            remaining -= exec_vol
        
        return ExecutionResult(cost=cost, volume_executed=volume - remaining)

    def _validate_order(
        self, 
        execution: ExecutionResult, 
        arrival_price: float
    ) -> OrderValidation:
        """Função pura: valida a execução contra limites de slippage"""
        # Caso especial: wait action (volume_executed = 0) é válido
        if execution.volume_executed <= 0:
            return OrderValidation(
                is_valid=True,
                avg_price=arrival_price,
                slippage=0.0,
                rejection_reason=""
            )
        
        avg_price = execution.cost / execution.volume_executed
        slippage = avg_price - arrival_price
        price_limit = arrival_price * (1 + self.max_slippage_pct)
        
        is_valid = avg_price <= price_limit
        rejection_reason = (
            f"Slippage {slippage:.6f} exceeds limit {self.max_slippage_pct * 100:.2f}%"
            if not is_valid else ""
        )
        
        return OrderValidation(
            is_valid=is_valid,
            avg_price=avg_price,
            slippage=slippage,
            rejection_reason=rejection_reason
        )

    def _calculate_reward(
        self, 
        validation: OrderValidation, 
        inventory_executed: float,
        is_terminal: bool,
        remaining_inventory: float,
        arrival_price: float
    ) -> float:
        """Função pura: calcula reward baseado em validação e estado"""
        match (validation.is_valid, inventory_executed > 0):
            case (False, _):
                return -0.5  # Penalidade por rejeição
            case (True, True):
                # Implementation Shortfall: penalidade proporcional ao slippage
                impl_shortfall = -validation.slippage * inventory_executed
                # Penalidade adicional se episódio termina com inventário restante
                opportunity_cost = (
                    -remaining_inventory * (arrival_price * 0.05)
                    if is_terminal and remaining_inventory > 0
                    else 0.0
                )
                return impl_shortfall + opportunity_cost
            case (True, False):
                return 0.0  # Nenhuma execução, mas válida (aguardando)
            case _:
                return 0.0

    def step(self, action: Literal[0, 1, 2, 3]) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Executa um passo do ambiente com pattern matching e funções puras"""
        row_b3 = self.lob_b3.iloc[self.current_step]
        row_base = self.lob_base.iloc[self.current_step]

        # Roteamento com pattern matching - Python 3.10+
        match action:
            case 0:  # Aguardar
                execution = ExecutionResult(cost=0.0, volume_executed=0.0)
            case 1:  # Roteia para B3
                execution = self._execute_order(100, row_b3)
            case 2:  # Roteia para Base Exchange
                execution = self._execute_order(100, row_base)
            case 3:  # Slicing (Agregação de Liquidez)
                exec_b3 = self._execute_order(100, row_b3)
                exec_base = self._execute_order(100, row_base)
                execution = ExecutionResult(
                    cost=exec_b3.cost + exec_base.cost,
                    volume_executed=exec_b3.volume_executed + exec_base.volume_executed
                )
            case _:
                raise ValueError(f"Invalid action: {action}")

        # Validação da ordem (função pura)
        validation = self._validate_order(execution, self.arrival_price)
        
        # Atualizar inventário apenas se validado
        inventory_executed = execution.volume_executed if validation.is_valid else 0.0
        self.inventory_remaining -= inventory_executed
        
        # Incrementar step
        self.current_step += 1
        done = (
            self.inventory_remaining <= 0 
            or self.current_step >= len(self.lob_b3) - 1
        )

        # Calcular reward (função pura)
        reward = self._calculate_reward(
            validation=validation,
            inventory_executed=inventory_executed,
            is_terminal=done,
            remaining_inventory=self.inventory_remaining,
            arrival_price=self.arrival_price
        )

        return self._get_obs(), reward, done, False, {'inventory_left': self.inventory_remaining}