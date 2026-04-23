import pytest
import pandas as pd
import numpy as np

# Importa a classe do ambiente
from src.sor_env import MultiVenueSOREnv

@pytest.fixture
def env():
    """
    Fixture do Pytest: Prepara os DataFrames mock de LOB para B3 e Base Exchange.
    Esse 'env' será injetado automaticamente em qualquer teste que o pedir como parâmetro.
    """
    # Cria LOB para B3 (5 timesteps, 5 níveis de profundidade)
    lob_b3_data = {
        'ask_1': [35.00, 35.01, 35.02, 34.99, 35.05],
        'vol_ask_1': [1500, 2000, 800, 3000, 500],
        'ask_2': [35.01, 35.02, 35.03, 35.00, 35.06],
        'vol_ask_2': [1000, 1500, 1200, 2500, 700],
        'ask_3': [35.02, 35.03, 35.04, 35.01, 35.07],
        'vol_ask_3': [800, 1000, 1000, 2000, 600],
        'ask_4': [35.03, 35.04, 35.05, 35.02, 35.08],
        'vol_ask_4': [600, 800, 800, 1500, 500],
        'ask_5': [35.04, 35.05, 35.06, 35.03, 35.09],
        'vol_ask_5': [500, 600, 600, 1000, 400],
    }
    lob_b3 = pd.DataFrame(lob_b3_data)

    # Cria LOB para Base Exchange (5 timesteps, 5 níveis de profundidade)
    lob_base_data = {
        'ask_1': [35.02, 35.03, 35.05, 35.01, 35.06],
        'vol_ask_1': [1200, 1800, 1000, 2800, 600],
        'ask_2': [35.03, 35.04, 35.06, 35.02, 35.07],
        'vol_ask_2': [900, 1200, 900, 2200, 550],
        'ask_3': [35.04, 35.05, 35.07, 35.03, 35.08],
        'vol_ask_3': [700, 900, 800, 1800, 450],
        'ask_4': [35.05, 35.06, 35.08, 35.04, 35.09],
        'vol_ask_4': [500, 700, 700, 1300, 350],
        'ask_5': [35.06, 35.07, 35.09, 35.05, 35.10],
        'vol_ask_5': [400, 500, 500, 900, 250],
    }
    lob_base = pd.DataFrame(lob_base_data)

    # Instancia o ambiente com 1.000 ações para executar
    environment = MultiVenueSOREnv(lob_b3=lob_b3, lob_base=lob_base, total_inventory=1000)
    environment.reset()

    return environment

def test_reset_initial_state(env):
    """Testa se o reset configura o estado inicial corretamente."""
    obs, info = env.reset()

    assert obs.shape == (5,)
    assert info['inventory_left'] == 1000
    # arrival_price é o mínimo entre B3 (35.00) e Base (35.02)
    assert env.arrival_price == pytest.approx(35.00)

def test_step_action_wait(env):
    """Testa a Ação 0 (Esperar). Não deve consumir inventário mas retorna penalidade por não executar."""
    obs, reward, terminated, truncated, info = env.step(0)

    assert info['inventory_left'] == 1000
    # Execução válida com 0 volume = penalidade por não executar (caso True, False)
    # _calculate_reward retorna 0.0 para (True, False)
    assert reward == pytest.approx(0.0)
    assert not terminated
    assert env.current_step == 1

def test_step_action_buy_small(env):
    """Testa a Ação 1 (Comprar 100 na B3)."""
    obs, reward, terminated, truncated, info = env.step(1)

    assert info['inventory_left'] == 900
    # Calcula execução: 100 unidades na B3 com ask_1=35.00, vol=1500
    # custo = 100 * 35.00 = 3500.0, avg_price = 3500/100 = 35.00
    # slippage = 35.00 - 35.00 = 0.0
    # impl_shortfall = -0.0 * 100 = 0.0
    assert reward == pytest.approx(0.0)

def test_step_action_buy_large(env):
    """Testa a Ação 2 (Comprar 100 na Base Exchange)."""
    obs, reward, terminated, truncated, info = env.step(2)

    assert info['inventory_left'] == 900
    # Similar à ação 1, mas na Base Exchange
    assert reward == pytest.approx(0.0)

def test_episode_termination(env):
    """Testa se o episódio é truncado no fim dos dados com inventário restante."""
    # Mock tem 5 timesteps, so current_step vai de 0-4
    # Ação 1: compra 100, inventory = 900, current_step = 1
    obs, reward, terminated, truncated, info = env.step(1)
    assert info['inventory_left'] == 900
    assert not terminated
    
    # Ação 1: compra 100, inventory = 800, current_step = 2
    obs, reward, terminated, truncated, info = env.step(1)
    assert info['inventory_left'] == 800
    assert not terminated
    
    # Ação 3: compra 200 (100 + 100), inventory = 600, current_step = 3
    obs, reward, terminated, truncated, info = env.step(3)
    assert info['inventory_left'] == 600
    assert not terminated
    
    # Ação 3: compra 200, inventory = 400, current_step = 4
    obs, reward, terminated, truncated, info = env.step(3)
    assert info['inventory_left'] == 400
    # current_step é 4, len(lob_b3) - 1 = 4, então time limit -> truncated=True
    assert not terminated
    assert truncated

def test_episode_truncation(env):
    """Testa truncation quando alcança o fim dos dados sem zerar inventário."""
    # O mock tem 5 linhas (índices 0-4)
    # Após reset, current_step = 0
    # Após step 1: current_step = 1, done = 1 >= 4? No
    # Após step 2: current_step = 2, done = 2 >= 4? No
    # Após step 3: current_step = 3, done = 3 >= 4? No
    # Após step 4: current_step = 4, done = 4 >= 4? Yes
    for i in range(4):
        obs, reward, terminated, truncated, info = env.step(0)
        if i < 3:  # Primeiros 3 steps
            assert not terminated
            assert not truncated
    
    # Após o 4º step
    assert env.current_step == 4
    assert not terminated
    assert truncated