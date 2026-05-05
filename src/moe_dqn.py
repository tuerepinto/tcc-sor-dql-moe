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


class MoENetwork(nn.Module):
    """Arquitetura Mixture of Experts (MoE) usada como "cérebro" do DQN.

    Esta é a classe que os testes utilizam diretamente, com a assinatura
    `MoENetwork(input_dim=5, output_dim=3, num_experts=3)`.
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

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        # x é o estado do LOB. Shape esperado: (batch_size, input_dim)

        # Passo 1: o Gerente avalia o mercado e distribui os pesos
        gate_logits = self.gating_network(x)
        gate_weights = F.softmax(gate_logits, dim=-1)

        # Passo 2: todos os especialistas produzem seus Q-Values
        # Shape: (batch_size, num_experts, output_dim)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # Passo 3: ponderamos a saída de cada especialista pelo peso do Gerente
        # gate_weights -> (batch_size, num_experts, 1)
        # Passo 4: somamos ao longo da dimensão de especialistas
        # Shape final: (batch_size, output_dim)
        final_q_values = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)

        if not return_aux:
            return final_q_values

        # Auxiliar de load-balancing: minimizar -H(gate) equivale a maximizar entropia.
        eps = 1e-8
        entropy = -(gate_weights * (gate_weights + eps).log()).sum(dim=-1).mean()
        aux_loss = -entropy

        aux_info = {
            "gate_entropy": float(entropy.detach().cpu().item()),
            "gate_mean": gate_weights.detach().mean(dim=0).cpu(),
        }
        return final_q_values, aux_loss, aux_info


class SingleQuantileExpert(nn.Module):
    """Expert que produz quantis por ação.

    Saída: (B, action_dim, n_quantiles)
    """

    def __init__(self, input_dim: int, action_dim: int, n_quantiles: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.n_quantiles = n_quantiles
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim * n_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)  # (B, A*NQ)
        return out.view(x.size(0), self.action_dim, self.n_quantiles)


class MoEQuantileNetwork(nn.Module):
    """MoE para QR-DQN.

    Forward:
      - return_aux=False -> (B, action_dim, n_quantiles)
      - return_aux=True  -> (q, aux_loss, aux_info)
    """

    def __init__(self, input_dim: int = 5, action_dim: int = 4, num_experts: int = 3, n_quantiles: int = 51) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.action_dim = action_dim
        self.n_quantiles = n_quantiles

        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
        )
        self.experts = nn.ModuleList(
            [SingleQuantileExpert(input_dim, action_dim, n_quantiles) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        gate_logits = self.gating_network(x)          # (B, E)
        gate_weights = F.softmax(gate_logits, dim=-1) # (B, E)

        # (B, E, A, NQ)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # Mistura densa para manter consistência com o MoE do DQN padrão.
        q = torch.sum(gate_weights.view(-1, self.num_experts, 1, 1) * expert_outputs, dim=1)

        if not return_aux:
            return q

        eps = 1e-8
        entropy = -(gate_weights * (gate_weights + eps).log()).sum(dim=-1).mean()
        aux_loss = -entropy
        aux_info = {
            "gate_entropy": float(entropy.detach().cpu().item()),
            "gate_mean": gate_weights.detach().mean(dim=0).cpu(),
        }
        return q, aux_loss, aux_info
