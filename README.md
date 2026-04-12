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

Principais Funcionalidades
--------------------------

- Reconstrução e normalização de snapshots do LOB a partir de dados de ticks.
- Ambiente de simulação financeira compatível com a API padrão de `gymnasium.Env`.
- Implementação de DQN com Experience Replay e Target Network.
- Benchmark contra estratégias rule-based (TWAP/VWAP).

Stack Tecnológico
-----------------

- Python 3.10+
- PyTorch (Deep Learning)
- Gymnasium (Reinforcement Learning)
- Polars / Pandas / NumPy (manipulação de dados)
- Matplotlib / Seaborn (visualização)

Requisitos
----------

As dependências principais estão listadas em `requirements.txt`. Para uso básico:

- Python 3.10 ou superior
- `torch>=2.2.2`
- `gymnasium` (se aplicável ao ambiente)

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

Como ainda não há scripts Python versionados neste repositório (por exemplo, arquivos como `train_*.py` ou `evaluate_*.py`), os comandos abaixo são apenas um modelo de uso.

Após criar os scripts de treinamento e avaliação, adapte os comandos, por exemplo:

- Treinamento do agente DQN:

   python train_<nome_do_script>.py

- Avaliação / benchmark contra TWAP/VWAP:

   python evaluate_<nome_do_script>.py

Se você estiver utilizando notebooks Jupyter, certifique-se de ativar o ambiente virtual antes de abrir o Jupyter Kernel.

Estrutura do Projeto
--------------------

Uma estrutura típica pode incluir (nomes meramente ilustrativos):

- `env/` – implementação do ambiente Gymnasium da B3/LOB.
- `models/` – arquiteturas DQN, MoE e utilitários de rede neural.
- `data/` – dados de mercado (não versionados ou com exemplos sintéticos).
- `scripts/` – scripts de treinamento, avaliação e pré-processamento.

Consulte o próprio repositório para a estrutura final.

Aviso
-----

Este projeto é estritamente acadêmico e voltado à pesquisa em microestrutura de mercado. Os modelos aqui treinados **não** constituem recomendação de investimento nem devem ser utilizados em produção (dinheiro real) sem validações adequadas de risco, compliance e auditoria independente.

Licença
-------

Este projeto é disponibilizado sob uma Licença de Uso Acadêmico. Para detalhes completos sobre permissões e restrições (incluindo proibição de uso comercial), consulte o arquivo `LICENSE.md` na raiz do repositório.