from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class AblationConfig:
    model_name: str
    device: str
    prompt_path: Path
    output_csv: Path
    output_plot: Path
    ablation_types: list[str]
    residual_position: str = "final"
    max_examples: Optional[int] = None


def load_config(path: str | Path) -> AblationConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    root = path.parent.parent
    return AblationConfig(
        model_name=raw["model_name"],
        device=raw.get("device", "auto"),
        prompt_path=_resolve(root, raw["prompt_path"]),
        output_csv=_resolve(root, raw["output_csv"]),
        output_plot=_resolve(root, raw["output_plot"]),
        ablation_types=list(raw.get("ablation_types", [])),
        residual_position=raw.get("residual_position", "final"),
        max_examples=raw.get("max_examples"),
    )


def _resolve(root: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else root / path

@dataclass
class InductionExample:
    prompt: str
    answer: str
    repeat_position: int = None