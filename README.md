# SMART ORDER ROUTER (SOR) INTELIGENTE PARA O MERCADO DE CAPITAIS BRASILEIRO

Projeto acadêmico de otimização de execução de ordens institucionais com Deep Q-Learning e Mixture of Experts, aplicado à microestrutura do mercado brasileiro.

## Visão Geral

Este repositório implementa um ambiente de simulação de mercado e um pipeline de treinamento/avaliação para um agente de roteamento de ordens baseado em Reinforcement Learning. O foco é aprender a decidir como fragmentar e enviar ordens em um ambiente de livro de ordens (LOB) com informação de liquidez, spread e dinâmica de execução.

O trabalho combina:

- ambiente de simulação do mercado em Gymnasium;
- rede neural do tipo Mixture of Experts (MoE);
- algoritmo de aprendizado Deep Q-Learning;
- suporte a variantes como QR-DQN;
- avaliação de desempenho contra baselines e replay de execução.

## Estrutura Atual do Repositório

```text
tcc-sor-dql-moe/
├── LICENSE.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── train_12m.yaml
│   └── train_synth.yaml
├── data/
│   └── l2_parquet/
│       └── venue=...
├── logs/
│   └── rewards_synthetic.csv
├── models/
│   ├── moe_dqn_sor.pth
│   ├── moe_dqn_sor_ITUB4_12m.pth
│   ├── moe_dqn_sor_PETR4_12m.pth
│   ├── moe_dqn_sor_SYNTHETIC.pth
│   └── moe_dqn_sor_VALE3_12m.pth
├── notebooks/
│   ├── 01_exploracao_lob.ipynb
│   ├── 02_train_agent.ipynb
│   └── 03_avaliacao_baselines.ipynb
├── scripts/
│   ├── bootstrap_l2_parquet.py
│   ├── build_l2_dataset.py
│   ├── eval_replay.py
│   ├── online_runner.py
│   ├── train_offline_12m.py
│   ├── train_synth.py
│   └── validate_l2_dataset.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── l2_dataset.py
│   │   ├── offline_dataset.py
│   │   └── paths.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── factory.py
│   │   ├── sor_env_numpy.py
│   │   └── sor_env_parquet.py
│   ├── eval/
│   │   ├── __init__.py
│   │   └── replay.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_io.py
│   │   └── moe_network.py
│   └── trainers/
│       ├── __init__.py
│       ├── dqn_runner.py
│       ├── qr_loss.py
│       └── train_agent.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_evaluate_baselines.py
│   ├── test_moe_dqn.py
│   └── test_sor_env.py
└── .gitignore
```

## Componentes Principais

### Ambiente de mercado

A lógica do ambiente fica em:

- `src/envs/factory.py`
- `src/envs/sor_env_numpy.py`
- `src/envs/sor_env_parquet.py`

Esses módulos definem a criação do ambiente, tanto para dados NumPy quanto para dados em parquet do dataset L2.

### Dados e preprocessamento

A preparação e leitura de dados estão em:

- `src/data/paths.py`
- `src/data/l2_dataset.py`
- `src/data/offline_dataset.py`
- `scripts/build_l2_dataset.py`
- `scripts/bootstrap_l2_parquet.py`
- `scripts/validate_l2_dataset.py`

### Modelo e treinamento

Os blocos do agente e do treinamento estão em:

- `src/models/moe_network.py`
- `src/models/model_io.py`
- `src/trainers/dqn_runner.py`
- `src/trainers/train_agent.py`
- `src/trainers/qr_loss.py`

### Avaliação

A avaliação de replay e baseline está concentrada em:

- `src/eval/replay.py`
- `scripts/eval_replay.py`
- `scripts/online_runner.py`

## Fluxo de Trabalho

### 1) Preparar dados

O projeto inclui scripts para montar e validar o conjunto de dados L2:

```bash
python scripts/bootstrap_l2_parquet.py
python scripts/validate_l2_dataset.py
```

### 2) Treinar o agente

Os treinamentos oficiais ficam em scripts na pasta `scripts/`:

```bash
python scripts/train_offline_12m.py
python scripts/train_synth.py
```

A rotina em `src/trainers/dqn_runner.py` coordena a execução do treino e salva checkpoints em `models/`.

### 3) Avaliar o agente

Para avaliar o modelo em um cenário de replay:

```bash
python scripts/eval_replay.py
```

### 4) Executar notebooks

Os notebooks de exploração e análise estão em `notebooks/`:

- `01_exploracao_lob.ipynb`
- `02_train_agent.ipynb`
- `03_avaliacao_baselines.ipynb`

## Requisitos

- Python 3.10+
- PyTorch
- Gymnasium
- NumPy / Pandas / PyArrow
- pytest

As dependências do projeto estão em `requirements.txt`.

## Instalação

```bash
git clone https://github.com/tuerepinto/tcc-sor-dql-moe.git
cd tcc-sor-dql-moe
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execução Rápida

### Testes

```bash
python -m pytest tests -q
```

### Treinamento do agente

```bash
python scripts/train_offline_12m.py
```

### Avaliação

```bash
python scripts/eval_replay.py
```

## Observações

- Este projeto é de natureza acadêmica e experimental.
- O treinamento e a avaliação dependem da disponibilidade dos dados em `data/l2_parquet`.
- Os checkpoints gerados ficam na pasta `models/` e podem ser reutilizados em execuções posteriores.
- O projeto usa uma estrutura modular em `src/`, em vez de uma única pasta com scripts no nível raiz.

## Licença

Este repositório é distribuído sob os termos da licença presente em `LICENSE.md`.