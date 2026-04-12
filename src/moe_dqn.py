import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleExpert(nn.Module):
    """Rede neural "especialista" individual usada dentro do MoE.
    No Mixture of Experts teremos várias instâncias desta classe, cada uma
    potencialmente se especializando em um regime de mercado distinto.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Expert(nn.Module):
    """Arquitetura Mixture of Experts (MoE) usada como "cérebro" do DQN.

    Esta é a classe que os testes utilizam diretamente, com a assinatura
    `Expert(input_dim=5, output_dim=3, num_experts=3)`.
    """

    def __init__(self, input_dim: int = 5, output_dim: int = 3, num_experts: int = 3) -> None:
        super().__init__()
        self.num_experts = num_experts

        # 1. Gating Network (o "Gerente")
        # Recebe o estado do mercado e produz um peso para cada especialista
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
        )

        # 2. Os Especialistas (Experts)
        # Usamos nn.ModuleList para registrar as redes no PyTorch corretamente
        self.experts = nn.ModuleList(
            [SingleExpert(input_dim, output_dim) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x é o estado do LOB. Shape esperado: (batch_size, input_dim)

        # Passo 1: o Gerente avalia o mercado e distribui os pesos
        gate_logits = self.gating_network(x)
        gate_weights = F.softmax(gate_logits, dim=-1)

        # Passo 2: todos os especialistas produzem seus Q-Values
        # Shape: (batch_size, num_experts, output_dim)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # Passo 3: ponderamos a saída de cada especialista pelo peso do Gerente
        # gate_weights -> (batch_size, num_experts, 1)
        weighted_expert_outputs = gate_weights.unsqueeze(-1) * expert_outputs

        # Passo 4: somamos ao longo da dimensão de especialistas
        # Shape final: (batch_size, output_dim)
        final_q_values = torch.sum(weighted_expert_outputs, dim=1)

        return final_q_values
