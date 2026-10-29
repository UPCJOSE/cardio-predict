"""Ejecuta preprocesamiento → entrenamiento → evaluación."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    """Un solo comando: descarga → limpia → entrena → evalúa."""
    runpy.run_path(str(ROOT / "scripts" / "download_data.py"), run_name="__main__")
    from src.data_prep import run as prep
    from src.evaluate import run as evaluate
    from src.train import run as train

    print("\n--- Preprocesamiento ---")
    prep()
    print("\n--- Entrenamiento ---")
    train()
    print("\n--- Evaluación ---")
    evaluate()
    print("\nListo. Métricas en reports/metrics.json y figuras en reports/figures/.")


if __name__ == "__main__":
    main()
