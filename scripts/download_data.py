"""Descarga el dataset público de indicadores cardíacos (BRFSS cleaned)."""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_RAW, RAW_CSV, RAW_URL  # noqa: E402


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    if RAW_CSV.exists() and RAW_CSV.stat().st_size > 1_000_000:
        print(f"Ya existe: {RAW_CSV} ({RAW_CSV.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"Descargando {RAW_URL}")
    urllib.request.urlretrieve(RAW_URL, RAW_CSV)
    print(f"Guardado en {RAW_CSV} ({RAW_CSV.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
