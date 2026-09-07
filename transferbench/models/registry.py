"""Stable family aliases; provider calls and credentials remain Inspect's concern."""

import math
import os
import re
from pathlib import Path

import yaml
from inspect_ai.model import Model, get_model
from pydantic import Field, model_validator

from transferbench.tasks.schema import Record


class Pricing(Record):
    input_per_million: float = Field(ge=0, allow_inf_nan=False)
    output_per_million: float = Field(ge=0, allow_inf_nan=False)

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_million + output_tokens * self.output_per_million
        ) / 1_000_000


class ModelSpec(Record):
    alias: str
    provider: str
    model: str
    family: str
    pricing: Pricing
    max_input_tokens: int = Field(default=32768, ge=1024)
    supports_seed: bool = False
    base_url: str | None = None
    version_pinned: bool = False

    @model_validator(mode="after")
    def real_pricing(self) -> "ModelSpec":
        if self.provider != "fake" and (
            self.pricing.input_per_million <= 0 or self.pricing.output_per_million <= 0
        ):
            raise ValueError(
                "Real models require positive conservative token prices, including self-hosted estimates"
            )
        if self.provider == "fake" and self.model not in {"vulnerable", "resistant"}:
            raise ValueError("fake model must be vulnerable or resistant")
        if not self.model.strip() or "${" in self.model:
            raise ValueError("model version must be resolved")
        return self

    @property
    def inspect_id(self) -> str:
        return f"{self.provider}/{self.model}"

    @property
    def simulated(self) -> bool:
        return self.provider == "fake"

    def resolve(self) -> Model:
        if self.simulated:
            from transferbench.models.fake import get_fake_model

            return get_fake_model(self.model)
        return get_model(self.inspect_id, base_url=self.base_url)


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader: UniqueLoader, node: yaml.MappingNode) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def read_yaml(path: Path) -> dict:
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def _expand(value):
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match) -> str:
        name = match.group(1)
        resolved = os.environ.get(name)
        if not resolved:
            raise ValueError(f"Required environment variable {name} is not set")
        return resolved

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, value)


class ModelRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        config = read_yaml(self.path)
        if set(config) != {"models"} or not isinstance(config["models"], dict):
            raise ValueError("Model registry must contain exactly a models mapping")
        self.entries = config["models"]

    def get(self, alias: str) -> ModelSpec:
        if alias not in self.entries:
            raise ValueError(f"Unknown model alias {alias!r}; choose from {sorted(self.entries)}")
        data = _expand(self.entries[alias])
        if not isinstance(data, dict):
            raise ValueError(f"Model alias {alias!r} must contain a configuration mapping")
        data.setdefault("family", alias)
        data["alias"] = alias
        return ModelSpec.model_validate(data)


def finite_cost(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError("cost must be a finite nonnegative number")
    return value
