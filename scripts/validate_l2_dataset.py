from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.paths import find_project_root, find_data_root
from src.data.l2_dataset import safe_dates, has_symbol, list_venues

SYMBOLS = ["PETR4", "VALE3", "ITUB4"]
VENUE_B3 = "B3"
VENUE_BASE = "BASE"

def main():
    root = find_project_root(Path.cwd())
    data_root = find_data_root(root)

    print("DATA_ROOT:", data_root)
    print("VENUES:", list_venues(data_root))

    ok = True
    for sym in SYMBOLS:
        e1 = has_symbol(data_root, VENUE_B3, sym)
        e2 = has_symbol(data_root, VENUE_BASE, sym)
        d1 = safe_dates(data_root, VENUE_B3, sym)
        d2 = safe_dates(data_root, VENUE_BASE, sym)
        common = sorted(set(d1) & set(d2))
        print(f"{sym} | B3={e1}({len(d1)} dias) BASE={e2}({len(d2)} dias) comum={len(common)}")
        if not common:
            ok = False

    if not ok:
        raise SystemExit("[FAIL] Dataset inválido para treino 12m.")
    print("[OK] Dataset validado.")

if __name__ == "__main__":
    main()