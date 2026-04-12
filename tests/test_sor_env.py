import pytest
import pandas as pd
import numpy as np

# Importa a classe do ambiente
from src.sor_env import B3LimitOrderBookEnv

@pytest.fixture
def env():
    """
    Fixture do Pytest: Prepara o DataFrame mock e instancia o ambiente.
    Esse 'env' será injetado automaticamente em qualquer teste que o pedir como parâmetro.
    """
    dados_mock = {
        'bid': [35.00, 35.01, 35.02, 34.99, 35.05],
        'ask': [35.02, 35.03, 35.05, 35.01, 35.06],
        'volume': [1500, 2000, 800, 3000, 500]
    }
    df_mock = pd.DataFrame(dados_mock)

    # Instancia o ambiente com 1.000 ações para executar
    environment = B3LimitOrderBookEnv(df_dados=df_mock, total_inventory=1000)
    environment.reset()

    return environment

def test_reset_initial_state(env):
    """Testa se o reset configura o estado inicial corretamente."""
    obs, info = env.reset()

    assert obs.shape == (5,)
    assert info['inventory_left'] == 1000
    # pytest.approx lida com imprecisões de ponto flutuante (ex: 35.010000000000005)
    assert env.arrival_price == pytest.approx(35.01)

def test_step_action_wait(env):
    """Testa a Ação 0 (Esperar). Não deve consumir inventário e deve dar penalidade leve."""
    obs, reward, terminated, truncated, info = env.step(0)

    assert info['inventory_left'] == 1000
    assert reward == -0.01
    assert not terminated
    assert env.current_step == 1

def test_step_action_buy_small(env):
    """Testa a Ação 1 (Comprar Lote Pequeno - 100 ações)."""
    obs, reward, terminated, truncated, info = env.step(1)

    assert info['inventory_left'] == 900
    # Custo = (35.02 - 35.01) * 100 = 1.0 -> Recompensa -1.0
    assert reward == pytest.approx(-1.0)

def test_step_action_buy_large(env):
    """Testa a Ação 2 (Comprar Lote Grande - 500 ações)."""
    obs, reward, terminated, truncated, info = env.step(2)

    assert info['inventory_left'] == 500
    # Custo = (35.02 - 35.01) * 500 = 5.0 -> Recompensa -5.0
    assert reward == pytest.approx(-5.0)

def test_episode_termination(env):
    """Testa se o episódio termina (terminated=True) quando o inventário zera."""
    env.step(2) # Compra 500
    obs, reward, terminated, truncated, info = env.step(2) # Compra mais 500

    assert info['inventory_left'] == 0
    assert terminated

def test_episode_truncation(env):
    """Testa se o episódio é interrompido (truncated=True) ao fim dos dados."""
    # O mock tem 5 linhas. O max_steps é 4.
    for _ in range(4):
        obs, reward, terminated, truncated, info = env.step(0)

    assert truncated
    assert not terminated