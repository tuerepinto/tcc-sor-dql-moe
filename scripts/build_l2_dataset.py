from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    out = Path("data/l2")
    out.mkdir(parents=True, exist_ok=True)
    print("Este script é o ponto oficial para construir/popular dataset L2 particionado.")
    print("Formato esperado: data/l2/venue=.../symbol=.../date=.../*.parquet")
    print("Implementar aqui ingestão real da sua fonte.")

if __name__ == "__main__":
    main()