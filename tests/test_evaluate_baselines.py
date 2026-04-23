import pytest
import pandas as pd
import numpy as np
import torch

# Importando as classes e funções do seu projeto
from src.sor_env import MultiVenueSOREnv
from src.moe_dqn import MoENetwork
from src.evaluate_baselines import simulate_twap, evaluate_agent

class TestBaselinesEvaluation:
    """
    Classe de testes para garantir que os baselines (TWAP) e a avaliação da IA
    estão calculando o preço médio e o volume executado corretamente.
    """

    @pytest.fixture
    def mock_market_data(self):
        """
        Fixture do Pytest que cria um LOB (Limit Order Book) pequeno e controlado
        para a B3 e para a Base Exchange.
        """
        steps = 10
        # Criando um cenário onde o preço é fixo para facilitar a validação matemática
        df_b3 = pd.DataFrame({
            'ask_1': [10.0] * steps, 'vol_ask_1': [100] * steps,
            'ask_2': [10.1] * steps, 'vol_ask_2': [100] * steps,
            'ask_3': [10.2] * steps, 'vol_ask_3': [100] * steps,
            'ask_4': [10.3] * steps, 'vol_ask_4': [100] * steps,
            'ask_5': [10.4] * steps, 'vol_ask_5': [100] * steps,
        })

        df_base = pd.DataFrame({
            'ask_1': [10.5] * steps, 'vol_ask_1': [100] * steps,
            'ask_2': [10.6] * steps, 'vol_ask_2': [100] * steps,
            'ask_3': [10.7] * steps, 'vol_ask_3': [100] * steps,
            'ask_4': [10.8] * steps, 'vol_ask_4': [100] * steps,
            'ask_5': [10.9] * steps, 'vol_ask_5': [100] * steps,
        })

        return df_b3, df_base

    @pytest.fixture
    def env(self, mock_market_data):
        """Fixture que inicializa o ambiente com os dados mockados."""
        df_b3, df_base = mock_market_data
        # Inventário de 500 ações para ser executado em 10 steps
        return MultiVenueSOREnv(df_b3, df_base, total_inventory=500)

    @pytest.fixture
    def untrained_model(self):
        """Fixture que inicializa a rede neural MoE-DQN."""
        return MoENetwork(input_dim=5, output_dim=4, num_experts=3)

    def test_simulate_twap_execution(self, env):
        """
        Testa se o algoritmo TWAP divide a ordem corretamente e calcula o preço médio.
        """
        total_inventory = 500
        preco_medio, vol_executado, meta = simulate_twap(env, total_inventory)

        # O TWAP deve tentar executar todo o inventário se houver liquidez
        assert vol_executado > 0
        assert vol_executado <= total_inventory

        # Como o TWAP roteia só para a B3 e o nosso mock da B3 tem ask_1 = 10.0,
        # o preço médio deve ser exatamente 10.0 (pois há 100 de volume por step e o TWAP pedirá 500/9 = 55 por step)
        assert preco_medio == pytest.approx(10.0, rel=1e-2)
        assert isinstance(meta, dict)
        assert "rejects" in meta

    def test_evaluate_agent_execution(self, env, untrained_model):
        """
        Testa se a função de avaliação da IA consegue rodar um episódio completo
        sem quebrar e retorna valores válidos de preço e volume.
        """
        preco_medio, vol_executado, meta = evaluate_agent(untrained_model, env)

        # Verifica se a IA executou algum volume (mesmo sendo um modelo não treinado, 
        # ele vai tomar ações aleatórias baseadas nos pesos iniciais da rede)
        assert isinstance(vol_executado, (int, float))
        assert vol_executado >= 0

        # O preço médio deve ser um número real (float) e maior ou igual a zero
        assert isinstance(preco_medio, float)
        assert preco_medio >= 0.0
        assert isinstance(meta, dict)
        assert "rejects" in meta

    def test_twap_vs_agent_format(self, env, untrained_model):
        """
        Garante que ambas as funções retornam a mesma estrutura de dados
        para que a comparação no script principal não quebre.
        """
        total_inventory = 100
        env.total_inventory = total_inventory

        res_twap = simulate_twap(env, total_inventory)
        res_ia = evaluate_agent(untrained_model, env)

        assert len(res_twap) == 3
        assert len(res_ia) == 3

        assert isinstance(res_twap[0], float) # Preço
        assert isinstance(res_ia[0], float)   # Preço
        assert isinstance(res_twap[2], dict)
        assert isinstance(res_ia[2], dict)