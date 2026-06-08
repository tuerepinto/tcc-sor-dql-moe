from pathlib import Path
from src.data.offline_dataset import list_available_dates

def safe_dates(root: Path, venue: str, symbol: str):
    try:
        return list_available_dates(root, venue, symbol)
    except FileNotFoundError:
        return []

def list_venues(root: Path):
    return sorted([p.name.replace("venue=", "") for p in root.glob("venue=*") if p.is_dir()])

def has_symbol(root: Path, venue: str, symbol: str) -> bool:
    return (root / f"venue={venue}" / f"symbol={symbol}").exists()