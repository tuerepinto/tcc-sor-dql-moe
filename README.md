SMART ORDER ROUTER (SOR) INTELIGENTE PARA O MERCADO DE CAPITAIS BRASILEIRO: UMA ABORDAGEM BASEADA EM DEEP Q-LEARNING E MIXTURE OF EXPERTS
================================================

Otimização da execução de ordens institucionais com Deep Q-Learning (DQN) e Mixture of Experts (MoE), aplicado à microestrutura da B3.

Este repositório contém o código-fonte e o ambiente de simulação desenvolvidos para um Trabalho de Conclusão de Curso (TCC) em Ciência de Dados / Inteligência Artificial aplicada a Finanças Quantitativas.

Visão Geral
-----------

O objetivo do projeto é substituir algoritmos estáticos tradicionais de roteamento de ordens (por exemplo, TWAP e VWAP) por um agente autônomo baseado em Deep Reinforcement Learning.

Focado na microestrutura do mercado de capitais brasileiro (B3) e na fragmentação de liquidez, o agente aprende a fracionar e enviar grandes ordens institucionais minimizando:

- Implementation Shortfall
- Slippage
- Impacto de mercado

Arquitetura
-----------

O projeto integra duas frentes principais de IA e modelagem de mercado:

- **Ambiente customizado (Gymnasium)**: MDP que simula a dinâmica de alta frequência do Limit Order Book (LOB) Nível 2.
- **Mixture of Experts (MoE)**: rede neural com ativação esparsa (gating network) que identifica o regime de mercado (alta volatilidade, baixa liquidez etc.) e aciona especialistas específicos para processar o estado do LOB.
- **Deep Q-Learning (DQN)**: o agente consome a saída da MoE para calcular Q-values e decidir ações como agredir o book, postar ordem passiva ou aguardar.

Como Funciona
-----------

1. **Treinamento** (`train_agent.py`):
   - Carrega dados do mercado B3
   - Instancia o `SOREnv` (ambiente de simulação do Limit Order Book)
   - Treina o agente DQN/MoE minimizando Implementation Shortfall
   - Salva os pesos treinados em `models/moe_dqn_sor.pth`

2. **Avaliação** (`evaluate_baselines.py` e `run_eval.py`):
   - Carrega o modelo treinado
   - Compara desempenho contra baselines (TWAP, VWAP, etc.)
   - Calcula métricas: Slippage, Impact, Execução média
   - Gera relatórios de comparação

3. **Análise** (Notebooks):
   - `01_exploracao_lob.ipynb`: Exploração inicial do Limit Order Book
   - `02_train_agent.ipynb`: Treinamento interativo do agente
   - `03_avaliacao_baselines.ipynb`: Visualização de resultados de avaliação

Principais Funcionalidades
--------------------------

- **Reconstrução do LOB**: Normalização de snapshots do Limit Order Book a partir de dados de ticks.
- **Ambiente de Simulação**: MDP (Markov Decision Process) compatível com `gymnasium.Env` que simula dinâmica de alta frequência do LOB.
- **Rede MoE**: Mixture of Experts com gating network para identificação de regimes de mercado e ativação esparsa.
- **Deep Q-Learning**: DQN com Experience Replay, Target Network e suporte a QR-DQN (Quantile Regression).
- **Benchmark**: Avaliação comparativa contra estratégias rule-based como TWAP, VWAP e outras baselines.
- **Testes Unitários**: Cobertura de testes com pytest para ambiente, modelo e funções de loss.

Stack Tecnológico
-----------------

- Python 3.12.12
- PyTorch (Deep Learning)
- Gymnasium (Reinforcement Learning)
- Polars / Pandas / NumPy (manipulação de dados)
- Matplotlib / Seaborn (visualização)
- pytest (testes)

Requisitos
----------

As dependências estão listadas em `requirements.txt`. Versões principais:

- Python 3.12 ou superior
- `torch>=2.2.2`
- `gymnasium`
- `numpy`, `pandas`, `polars`
- `pytest` (para executar testes)

Instalação
----------

1. Clone o repositório:

   git clone https://github.com/tuerepinto/tcc-sor-dql-moe.git
   cd tcc-sor-dql-moe

2. Crie e ative um ambiente virtual (opcional, mas recomendado):

   python -m venv .venv
   source .venv/bin/activate

3. Instale as dependências:

   pip install -r requirements.txt

Uso
---

**Treinamento do agente DQN/MoE:**

   /Users/tuerepinto/Documents/repository/tcc-sor-dql-moe/.venv/bin/python src/train_agent.py

**Avaliação / benchmark contra baselines (TWAP/VWAP):**

   /Users/tuerepinto/Documents/repository/tcc-sor-dql-moe/.venv/bin/python src/evaluate_baselines.py

Ou, para executar o script de avaliação completo:

   /Users/tuerepinto/Documents/repository/tcc-sor-dql-moe/.venv/bin/python run_eval.py

**Executar testes:**

   /Users/tuerepinto/Documents/repository/tcc-sor-dql-moe/.venv/bin/python -m pytest tests/

**Notebooks Jupyter:**

Certifique-se de ativar o ambiente virtual antes de abrir o Jupyter Kernel:

   source .venv/bin/activate
   jupyter notebook

Estrutura do Projeto
--------------------

Estrutura atual do repositório:

```
tcc-sor-dql-moe/
│
├── src/                        # Código-fonte principal
│   ├── __init__.py             # Torna 'src' um pacote Python
│   ├── sor_env.py              # Ambiente do Limit Order Book (B3)
│   ├── moe_dqn.py              # Arquitetura Mixture of Experts (MoE)
│   ├── qr_loss.py              # Implementação de QR-DQN loss
│   ├── wrappers.py             # Wrappers customizados para o ambiente
│   ├── train_agent.py          # Script de treinamento do agente DQN/MoE
│   └── evaluate_baselines.py   # Avaliação e benchmark contra estratégias
│
├── tests/                      # Testes unitários (pytest)
│   ├── conftest.py             # Configuração comum de testes
│   ├── test_sor_env.py         # Testes do ambiente B3LimitOrderBookEnv
│   ├── test_moe_dqn.py         # Testes da rede MoE (Expert)
│   └── test_evaluate_baselines.py  # Testes de avaliação de baselines
│
├── notebooks/                  # Notebooks Jupyter para exploração e demonstrações
│   ├── 01_exploracao_lob.ipynb
│   ├── 02_train_agent.ipynb
│   └── 03_avaliacao_baselines.ipynb
│
├── models/                     # Modelos treinados (pesos de rede neural)
│   └── moe_dqn_sor.pth         # Checkpoint do agente DQN/MoE treinado
│
├── data/                       # Arquivos de dados brutos ou pré-processados
│
├── run_eval.py                 # Script para execução completa de avaliação
├── requirements.txt            # Dependências do ambiente Python
├── README.md                   # Documentação principal do projeto
├── LICENSE.md                  # Licença de uso acadêmico
├── .gitignore                  # Arquivos/pastas ignorados pelo Git
└── .venv/                      # Ambiente virtual Python (não versionado)
```

Aviso
-----

Este projeto é estritamente acadêmico e voltado à pesquisa em microestrutura de mercado. Os modelos aqui treinados **não** constituem recomendação de investimento nem devem ser utilizados em produção (dinheiro real) sem validações adequadas de risco, compliance e auditoria independente.

Licença
-------

Este projeto é disponibilizado sob uma Licença de Uso Acadêmico. Para detalhes completos sobre permissões e restrições (incluindo proibição de uso comercial), consulte o arquivo `LICENSE.md` na raiz do repositório.