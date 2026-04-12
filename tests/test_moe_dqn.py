import pytest
import torch
import torch.nn.functional as F

# Importa a rede neural que criamos
from src.moe_dqn import Expert

@pytest.fixture
def model():
    """
    Fixture que instancia a rede MoE com os parâmetros padrão do nosso TCC:
    5 entradas (LOB + Inventário), 3 saídas (Ações) e 3 Especialistas.
    """
    return Expert(input_dim=5, output_dim=3, num_experts=3)

@pytest.fixture
def dummy_input():
    """
    Fixture que cria um 'lote' (batch) de dados simulados.
    Shape: (batch_size=4, input_dim=5)
    Simula 4 momentos diferentes do mercado passando pela rede ao mesmo tempo.
    """
    return torch.tensor([
        [35.00, 35.02, 1500, 2000, 1000], # Estado 1
        [34.90, 34.95, 5000, 1000, 900],  # Estado 2
        [35.10, 35.11, 200,  300,  500],  # Estado 3
        [35.05, 35.08, 1000, 1000, 100]   # Estado 4
    ], dtype=torch.float32)

def test_model_initialization(model):
    """Testa se a arquitetura foi montada com as dimensões e listas corretas."""
    assert model.num_experts == 3
    assert len(model.experts) == 3
    # Verifica se a primeira camada do primeiro especialista tem 5 entradas
    assert model.experts[0].network[0].in_features == 5

def test_forward_pass_shape(model, dummy_input):
    """
    Testa se a rede consegue processar os dados sem quebrar (crash) 
    e se a saída tem o formato correto: (batch_size, output_dim).
    """
    q_values = model(dummy_input)

    # Esperamos 4 linhas (batch_size) e 3 colunas (Q-Values para as 3 ações)
    assert q_values.shape == (4, 3)

def test_gating_network_softmax(model, dummy_input):
    """
    Testa a matemática do 'Gerente' (Gating Network).
    A soma dos pesos distribuídos para os especialistas DEVE ser exatamente 1.0.
    """
    # Passa os dados apenas no gerente
    gate_logits = model.gating_network(dummy_input)
    gate_weights = F.softmax(gate_logits, dim=-1)

    # Soma os pesos de cada linha (dim=1)
    sum_of_weights = torch.sum(gate_weights, dim=1)

    # Verifica se a soma de todas as 4 linhas é igual a 1.0
    # Usamos torch.allclose porque operações de ponto flutuante podem dar 0.9999999
    expected_sum = torch.ones(4, dtype=torch.float32)
    assert torch.allclose(sum_of_weights, expected_sum)

def test_backpropagation_connection(model, dummy_input):
    """
    O teste mais crítico para IA: Verifica se o grafo computacional está conectado.
    Se isso falhar, a rede nunca vai aprender (os pesos não serão atualizados).
    """
    # 1. Faz o forward pass
    q_values = model(dummy_input)

    # 2. Cria um alvo falso (target) e calcula uma perda (loss) fictícia
    dummy_target = torch.randn(4, 3)
    loss = F.mse_loss(q_values, dummy_target)

    # 3. Faz o backward pass (calcula os gradientes)
    loss.backward()

    # 4. Verifica se o gerente (gating network) recebeu gradientes
    assert model.gating_network[0].weight.grad is not None

    # 5. Verifica se o primeiro especialista recebeu gradientes
    assert model.experts[0].network[0].weight.grad is not None