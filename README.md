<div align="center">

# 🧠 Smart Order Router (SOR) Inteligente
### Uma Abordagem Baseada em Deep Q-Learning e Mixture of Experts para o Mercado de Capitais Brasileiro

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-brightgreen.svg)](https://gymnasium.farama.org/)
[![License: Academic](https://img.shields.io/badge/License-Academic-red.svg)](#licença)

*Trabalho de Conclusão de Curso (TCC) em Inteligência Artificial e Aprendizado de Máquina.*

[Visão Geral](#-visão-geral) • 
[Arquitetura](#-arquitetura-e-metodologia) • 
[Resultados](#-resultados-e-impacto-de-negócios) • 
[Instalação](#-instalação-e-uso) • 
[Estrutura](#-estrutura-do-projeto)

</div>

---

## 📖 Visão Geral

Com a iminente quebra do monopólio da B3 e a chegada de novas *venues* (como a Base Exchange), a fragmentação de liquidez no Brasil exigirá sistemas de roteamento de ordens extremamente adaptativos. Algoritmos estáticos tradicionais (como TWAP e VWAP) deixam "pegadas" no mercado e são alvos fáceis para robôs de alta frequência (HFTs).

Este projeto propõe a substituição dessas heurísticas por um **Agente Autônomo baseado em Deep Reinforcement Learning**. Focado na microestrutura do mercado, o agente aprende a fracionar e enviar grandes ordens institucionais de forma furtiva, minimizando:
- **Implementation Shortfall (IS)**
- **Slippage**
- **Impacto de Mercado**

---

## 🏗️ Arquitetura e Metodologia

O projeto integra engenharia de sistemas complexos com modelos de Inteligência Artificial no estado da arte, modelando o mercado financeiro como um Processo de Decisão de Markov (MDP).

### 1. Ambiente de Simulação (Gymnasium)
O motor de simulação consome dados de profundidade do *Limit Order Book* (Nível 2) em milissegundos, extraídos via **yfinance / IBKR API**.
- **Estado ($S_t$):** Tensor de 5 dimensões (Melhor Bid, Melhor Ask, Vol Bid, Vol Ask, Inventário Restante).
- **Ações ($A_t$):** (0) Aguardar, (1) Comprar Lote Pequeno, (2) Comprar Lote Grande.
- **Recompensa ($R_t$):** Penalidade financeira baseada no *Implementation Shortfall* real gerado pelo consumo de liquidez.

### 2. O Cérebro: Mixture of Experts (MoE) + DQN
Em vez de uma rede neural monolítica, o núcleo decisório utiliza a arquitetura **Mixture of Experts (MoE)** integrada ao **Deep Q-Network (DQN)**:
- Uma *Gating Network* identifica o regime de mercado (alta volatilidade, baixa liquidez).
- O estado é roteado dinamicamente para "Especialistas" neurais específicos.
- O agente calcula os *Q-Values* para decidir a ação ótima em frações de segundo, evitando o esquecimento catastrófico (*catastrophic forgetting*).

---

## 📊 Resultados e Impacto de Negócios

O modelo foi avaliado em um cenário *Out-of-Sample* simulando uma ordem institucional de 10.000 ações, comparado diretamente contra o *baseline* TWAP.

### Performance Quantitativa
| Métrica | TWAP (Baseline estático) | MoE-DQN (IA Adaptativa) |
| :--- | :--- | :--- |
| **Comportamento** | Cego à liquidez imediata | Furtivo e adaptativo |
| **Preço Médio** | Superior (Pior) | Inferior (Melhor) |
| **IS / Slippage** | Alto | Drasticamente reduzido |
| **Rejeições** | Frequentes em baixa liquidez | Mitigadas pela IA |

<details>
<summary><b>📈 Clique aqui para ver os Gráficos de Desempenho</b></summary>
<br>

*(Nota: Adicione as imagens geradas pelos notebooks na pasta `docs/` do seu repositório)*

**1. Curva de Aprendizado do Agente (Convergência)**
> O gráfico comprova que o agente aprende a otimizar a recompensa ao longo de 500 episódios.
> 
> `![Curva de Aprendizado](docs/learning_curve.png)`

**2. Comparação de Performance (TWAP vs MoE-DQN)**
> Redução do Implementation Shortfall e economia financeira total gerada.
> 
> `![Comparação de Performance](docs/performance_comparison.png)`

**3. Diagnóstico Microestrutural (Raio-X da IA)**
> Demonstração da *Gating Network* alternando especialistas conforme a liquidez do mercado.
> 
> `![Diagnóstico Microestrutural](docs/diagnostic_panels.png)`

</details>

### 💼 Viabilidade Econômica e Lei do Bem
Além da economia direta gerada pela redução de *slippage* (que pode representar dezenas de milhões de reais anualmente para uma mesa institucional), o caráter inovador da arquitetura MoE-DQN permite o enquadramento do projeto na **Lei do Bem (Lei nº 11.196/05)**. Isso transforma o custo de infraestrutura de IA e P&D em um investimento subsidiado por isenções fiscais, gerando um ROI altamente assimétrico.

---

## 🚀 Instalação e Uso

### Pré-requisitos
- Python 3.12+
- Gerenciador de pacotes `uv` (recomendado) ou `pip`

### Instalação
```bash
# Clone o repositório
git clone https://github.com/tuerepinto/tcc-sor-dql-moe.git
cd tcc-sor-dql-moe

# Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt
```

### Execução
**1. Treinamento do Agente:**
```bash
python src/train_agent.py
```

**2. Avaliação e Benchmark (TWAP vs IA):**
```bash
python src/evaluate_baselines.py
# ou execute o pipeline completo:
python run_eval.py
```

**3. Testes Unitários:**
```bash
python -m pytest tests/
```

---

## 📂 Estrutura do Projeto

```text
tcc-sor-dql-moe/
├── src/                      # Código-fonte principal
│   ├── sor_env.py            # Ambiente MDP do Limit Order Book
│   ├── moe_dqn.py            # Arquitetura Mixture of Experts (MoE)
│   ├── train_agent.py        # Pipeline de treinamento
│   └── evaluate_baselines.py # Benchmark contra TWAP/VWAP
├── notebooks/                # Análises e visualizações (Jupyter)
│   ├── 01_exploracao_lob.ipynb
│   ├── 02_train_agent.ipynb
│   └── 03_avaliacao_baselines.ipynb
├── tests/                    # Testes unitários (pytest)
├── models/                   # Checkpoints do modelo treinado (.pth)
├── data/                     # Datasets sintéticos e Tick Data
└── requirements.txt          # Dependências do projeto
```

---

## ⚠️ Aviso Legal

Este projeto é estritamente acadêmico e voltado à pesquisa em microestrutura de mercado. Os modelos aqui treinados **não constituem recomendação de investimento** nem devem ser utilizados em produção (dinheiro real) sem validações adequadas de risco, *compliance* (travas e *kill switches*) e auditoria independente.

## 📄 Licença

Este projeto é disponibilizado sob uma **Licença de Uso Acadêmico**. Para detalhes completos sobre permissões e restrições (incluindo a proibição de uso comercial não autorizado), consulte o arquivo `LICENSE.md` na raiz do repositório.

---
<div align="center">
  <i>Desenvolvido por <b>Tuerê Pinto</b> — 2026</i>
</div>
