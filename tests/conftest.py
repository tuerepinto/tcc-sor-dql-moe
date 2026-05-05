"""Configuração comum para os testes.

Garante que o diretório raiz do projeto (onde fica a pasta `src/`) esteja
no sys.path, permitindo imports do tipo `from src...` mesmo quando o
pytest é executado a partir de subdiretórios ou por IDEs.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Diretório raiz do projeto (um nível acima de tests/)
ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
