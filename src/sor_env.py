import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class B3LimitOrderBookEnv(gym.Env):
    
    """
    Ambiente de simulação de LOB da B3 para treinamento de Smart Order Router.
    Objetivo do Agente: Executar uma ordem de compra grande minimizando o custo (slippage).
    """

    def __init__(self, df_dados, total_inventory=10000):
        super().__init__()

        # Carrega os dados históricos (o CSV gerado no passo anterior)
        self.df = df_dados.reset_index(drop=True)
        self.max_steps = len(self.df) - 1

        # Configurações da Ordem Institucional
        self.initial_inventory = total_inventory
        self.inventory = self.initial_inventory
        self.current_step = 0

        # Preço de referência (Arrival Price) para calcular o Implementation Shortfall
        self.arrival_price = 0.0 

        # 1. ESPAÇO DE AÇÕES (Action Space)
        # 0 = Esperar (Não faz nada)
        # 1 = Comprar Lote Pequeno (ex: 100 ações)
        # 2 = Comprar Lote Grande (ex: 500 ações)
        self.action_space = spaces.Discrete(3)

        # 2. ESPAÇO DE OBSERVAÇÃO (Observation Space)
        # O que o robô "vê" a cada milissegundo: [Bid, Ask, Vol_Bid, Vol_Ask, Inventário_Restante]
        # Usamos Box para definir um array de 5 posições com valores contínuos (float)
        self.observation_space = spaces.Box(
            low=0.0, 
            high=np.inf, 
            shape=(5,), 
            dtype=np.float32
        )

    def _get_observation(self):
        """Função auxiliar para capturar a linha atual do LOB e o inventário."""
        row = self.df.iloc[self.current_step]
        obs = np.array([
            row['bid'], 
            row['ask'], 
            row['volume'], # Assumindo que seja o volume do topo do book
            row['volume'], # Simplificação para o exemplo
            self.inventory
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        """Reinicia o ambiente para o início do pregão/episódio."""
        super().reset(seed=seed)

        self.current_step = 0
        self.inventory = self.initial_inventory

        # Define o preço de chegada (benchmark) como o preço médio do primeiro tick
        first_row = self.df.iloc[0]
        self.arrival_price = (first_row['bid'] + first_row['ask']) / 2.0

        obs = self._get_observation()
        info = {"inventory_left": self.inventory}

        return obs, info

    def step(self, action):
        """Executa a ação do agente, calcula o custo e avança o tempo."""
        row = self.df.iloc[self.current_step]
        current_ask = row['ask'] # Preço que pagamos ao comprar a mercado

        executed_qty = 0
        reward = 0.0

        # Lógica de Execução baseada na Ação escolhida
        if action == 1:
            executed_qty = min(100, self.inventory) # Lote pequeno
        elif action == 2:
            executed_qty = min(500, self.inventory) # Lote grande

        # Se o agente decidiu comprar (ação 1 ou 2)
        if executed_qty > 0:
            self.inventory -= executed_qty

            # CÁLCULO DA RECOMPENSA (Implementation Shortfall)
            # Custo = Preço Executado - Preço de Chegada (Benchmark)
            # Como queremos MINIMIZAR o custo, a recompensa é o custo NEGATIVO.
            cost_per_share = current_ask - self.arrival_price
            trade_cost = cost_per_share * executed_qty

            reward = -trade_cost 
        else:
            # Se escolheu esperar (ação 0), damos uma micro penalidade 
            # para incentivar o robô a não demorar o dia todo para executar a ordem.
            reward = -0.01 

        # Avança o tempo (vai para a próxima linha do CSV)
        self.current_step += 1

        # Verifica se o episódio acabou
        terminated = bool(self.inventory <= 0) # Sucesso: executou toda a ordem
        truncated = bool(self.current_step >= self.max_steps) # Falha: acabou o dia/dados

        # Penalidade severa se o dia acabar e a ordem não for totalmente executada
        if truncated and self.inventory > 0:
            reward -= (self.inventory * current_ask * 0.01) # Penalidade de 1% sobre o saldo

        next_obs = self._get_observation()
        info = {
            "inventory_left": self.inventory,
            "execution_price": current_ask if executed_qty > 0 else 0
        }

        return next_obs, reward, terminated, truncated, info