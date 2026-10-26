"""Small, reproducible CNN prototype for three-scale K-line charts."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Sequence

from .charts import QueryCharts, build_query_charts
from .config import DEFAULT_CONFIG, ProjectConfig
from .market_data import MarketData
from .retrieval import RetrievalSample, SampleEmbeddings


CNN_MODEL_VERSION = "cnn-stage11-v1"


@dataclass(frozen=True, slots=True)
class CNNTrainingConfig:
    image_size: tuple[int, int] = (64, 64)
    embedding_dim: int = 128
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 1e-3
    temperature: float = 0.2
    seed: int = 20261026
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.image_size != (64, 64):
            raise ValueError("阶段十一 CNN 原型固定使用 64×64 图像")
        if self.embedding_dim != 128:
            raise ValueError("阶段十一 CNN 原型固定输出 128 维向量")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size 和 epochs 必须大于 0")
        if self.learning_rate <= 0 or self.temperature <= 0:
            raise ValueError("learning_rate 和 temperature 必须大于 0")


@dataclass(frozen=True, slots=True)
class SampleKey:
    symbol: str
    date: date

    def as_dict(self) -> dict[str, str]:
        return {"symbol": self.symbol, "date": self.date.isoformat()}


def _torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "CNN 原型需要可选依赖 PyTorch，请执行 .venv/bin/pip install -e '.[cnn]'"
        ) from exc
    return torch, nn, functional


def select_device(requested: str = "auto") -> str:
    torch, _, _ = _torch()
    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("当前 PyTorch 环境不可用 MPS")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("当前 PyTorch 环境不可用 CUDA")
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_network():
    _, nn, _ = _torch()

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
            )

        def forward(self, values):
            return self.head(self.features(values))

    return Network()


class CNNEmbedder:
    """Load a stage-11 checkpoint and encode one chart or three query charts."""

    dimension = 128

    def __init__(self, checkpoint: str | Path, *, device: str = "auto") -> None:
        torch, _, functional = _torch()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.device = select_device(device if device != "auto" else payload.get("device", "auto"))
        self.model = _build_network().to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.functional = functional
        self.model_version = payload.get("model_version", CNN_MODEL_VERSION)
        self.training_config = payload.get("training_config", {})

    def encode(self, image: object) -> tuple[float, ...]:
        torch, _, _ = _torch()
        values = _image_tensor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(values)
            embedding = self.functional.normalize(embedding, dim=1)[0]
        return tuple(float(value) for value in embedding.cpu().tolist())

    def encode_query_charts(self, charts: QueryCharts) -> SampleEmbeddings:
        return SampleEmbeddings(
            monthly=self.encode(charts.monthly),
            weekly=self.encode(charts.weekly),
            daily=self.encode(charts.daily),
        )


def build_cnn_sample_embedding(
    market_data: MarketData,
    symbol: str,
    trading_date: date | str,
    embedder: CNNEmbedder,
    *,
    config: ProjectConfig = DEFAULT_CONFIG,
) -> RetrievalSample:
    parsed_date = trading_date if isinstance(trading_date, date) else date.fromisoformat(trading_date)
    charts = build_query_charts(market_data, symbol, parsed_date, config)
    return RetrievalSample(
        symbol=symbol,
        date=parsed_date,
        embeddings=embedder.encode_query_charts(charts),
    )


def build_split_manifest(
    market_data: MarketData,
    symbols: Sequence[str],
    *,
    config: ProjectConfig = DEFAULT_CONFIG,
    max_per_symbol_split: int = 64,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
) -> dict[str, object]:
    """Create a deterministic chronological split without using future labels."""

    if not symbols:
        raise ValueError("至少需要一个训练股票")
    if max_per_symbol_split <= 0:
        raise ValueError("max_per_symbol_split 必须大于 0")
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("训练集和验证集比例必须为正，且总和小于 1")

    history_count = config.monthly_period * config.daily_window
    future_count = max(config.future_horizons)
    all_dates = sorted(
        {
            trading_date
            for symbol in symbols
            for trading_date in market_data.dates(symbol)[history_count - 1 :]
            if market_data.validate_sample(
                symbol,
                trading_date,
                history_count=history_count,
                future_count=future_count,
            ).valid
        }
    )
    if len(all_dates) < 10:
        raise ValueError("可用于 CNN 时间切分的有效交易日不足")
    train_cutoff = all_dates[max(0, int(len(all_dates) * train_ratio) - 1)]
    validation_cutoff = all_dates[max(0, int(len(all_dates) * (train_ratio + validation_ratio)) - 1)]

    splits: dict[str, list[SampleKey]] = {"train": [], "validation": [], "test": []}
    for symbol in symbols:
        valid_dates = [
            trading_date
            for trading_date in market_data.dates(symbol)[history_count - 1 :]
            if market_data.validate_sample(
                symbol,
                trading_date,
                history_count=history_count,
                future_count=future_count,
            ).valid
        ]
        buckets = {
            "train": [item for item in valid_dates if item <= train_cutoff],
            "validation": [item for item in valid_dates if train_cutoff < item <= validation_cutoff],
            "test": [item for item in valid_dates if item > validation_cutoff],
        }
        for name, values in buckets.items():
            splits[name].extend(
                SampleKey(symbol, trading_date)
                for trading_date in _evenly_spaced(values, max_per_symbol_split)
            )

    return {
        "model_version": CNN_MODEL_VERSION,
        "source_hash": market_data.source_hash,
        "config": {
            "monthly_period": config.monthly_period,
            "weekly_period": config.weekly_period,
            "daily_window": config.daily_window,
            "chart_size": list(config.chart_size),
            "future_horizons": list(config.future_horizons),
        },
        "cutoffs": {
            "train_last_date": train_cutoff.isoformat(),
            "validation_last_date": validation_cutoff.isoformat(),
        },
        "splits": {
            name: [item.as_dict() for item in values]
            for name, values in splits.items()
        },
    }


def write_split_manifest(manifest: dict[str, object], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def load_split_manifest(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def train_cnn(
    market_data: MarketData,
    manifest: dict[str, object],
    output_path: str | Path,
    *,
    config: CNNTrainingConfig = CNNTrainingConfig(),
) -> dict[str, object]:
    """Train a two-view contrastive CNN on the manifest's train split."""

    torch, _, functional = _torch()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    model = _build_network().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_keys = _manifest_keys(manifest, "train")
    if not train_keys:
        raise ValueError("训练切分为空")

    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(config.epochs):
        order = list(range(len(train_keys)))
        random.Random(config.seed + epoch).shuffle(order)
        losses: list[float] = []
        for start in range(0, len(order), config.batch_size):
            batch_keys = [train_keys[index] for index in order[start : start + config.batch_size]]
            base = torch.stack([_sample_tensor(market_data, key, DEFAULT_CONFIG) for key in batch_keys])
            first = _augment_tensor(base, torch, config.seed + epoch * 100_000 + start)
            second = _augment_tensor(base, torch, config.seed + epoch * 100_000 + start + 1)
            batch_size = base.shape[0]
            first = first.reshape(batch_size * 3, 1, 64, 64).to(device)
            second = second.reshape(batch_size * 3, 1, 64, 64).to(device)
            first_embedding = functional.normalize(model(first), dim=1)
            second_embedding = functional.normalize(model(second), dim=1)
            logits = first_embedding @ second_embedding.T / config.temperature
            labels = torch.arange(logits.shape[0], device=device)
            loss = (functional.cross_entropy(logits, labels) + functional.cross_entropy(logits.T, labels)) / 2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "loss": sum(losses) / len(losses)})

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_version": CNN_MODEL_VERSION,
        "source_hash": manifest.get("source_hash"),
        "training_config": asdict(config),
        "device": device,
        "model_state": model.state_dict(),
        "history": history,
        "manifest_hash": _json_hash(manifest),
    }
    torch.save(payload, output)
    return {
        "checkpoint": str(output),
        "model_version": CNN_MODEL_VERSION,
        "device": device,
        "train_samples": len(train_keys),
        "history": history,
        "manifest_hash": payload["manifest_hash"],
    }


def _manifest_keys(manifest: dict[str, object], split: str) -> list[SampleKey]:
    values = manifest.get("splits", {}).get(split, [])
    return [SampleKey(item["symbol"], date.fromisoformat(item["date"])) for item in values]


def _sample_tensor(market_data: MarketData, key: SampleKey, config: ProjectConfig):
    torch, _, _ = _torch()
    charts = build_query_charts(market_data, key.symbol, key.date, config)
    return torch.stack([_image_tensor(image) for image in (charts.monthly, charts.weekly, charts.daily)])


def _image_tensor(image: object):
    torch, _, _ = _torch()
    pixels = torch.tensor(getattr(image, "pixels"), dtype=torch.float32) / 255.0
    return (1.0 - pixels).unsqueeze(0)


def _augment_tensor(values, torch, seed: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    result = values.clone()
    if torch.rand((), generator=generator).item() < 0.5:
        result = torch.flip(result, dims=(-1,))
    shift_x = int(torch.randint(-2, 3, (), generator=generator))
    shift_y = int(torch.randint(-2, 3, (), generator=generator))
    result = torch.roll(result, shifts=(shift_y, shift_x), dims=(-2, -1))
    noise = torch.randn(result.shape, generator=generator) * 0.015
    return (result + noise).clamp(0.0, 1.0)


def _evenly_spaced(values: Sequence[date], limit: int) -> list[date]:
    if len(values) <= limit:
        return list(values)
    positions = [round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)]
    return [values[position] for position in positions]


def _json_hash(value: object) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

