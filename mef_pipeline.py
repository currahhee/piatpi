from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


BRACKET_IMAGE_PATTERN = re.compile(
    r"(?P<set_id>set\d+)_(?P<timestamp>\d{8}_\d{6})_(?P<label>[^_]+)_exp(?P<exposure>\d+)us\.(?P<ext>jpg|jpeg|png)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BracketImage:
    label: str
    path: Path
    requested_exposure_us: int | None = None
    actual_exposure_us: int | None = None
    actual_gain: float | None = None

    def load_bgr(self) -> np.ndarray:
        frame = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Unable to decode image: {self.path}")
        return frame


@dataclass(frozen=True)
class BracketSet:
    set_id: str
    timestamp: str
    directory: Path
    images: tuple[BracketImage, ...]
    metadata_path: Path | None = None
    bracket_mode: str | None = None
    auto_exposure_us: int | None = None
    auto_gain: float | None = None

    def labels(self) -> tuple[str, ...]:
        return tuple(image.label for image in self.images)

    def load_bgr_images(self) -> list[np.ndarray]:
        return [image.load_bgr() for image in self.images]


@dataclass(frozen=True)
class ProcessSummary:
    processed: int
    skipped: int
    output_paths: tuple[Path, ...]


class FusionBackend:
    name = "base"

    def fuse(self, images: Sequence[np.ndarray], bracket_set: BracketSet) -> np.ndarray:
        raise NotImplementedError


class OpenCvMertensBackend(FusionBackend):
    name = "opencv"

    def __init__(self) -> None:
        self._merge = cv2.createMergeMertens()

    def fuse(self, images: Sequence[np.ndarray], bracket_set: BracketSet) -> np.ndarray:
        if len(images) < 2:
            raise ValueError(
                f"Bracket set {bracket_set.set_id} needs at least 2 images for fusion"
            )

        float_images = [image.astype(np.float32) / 255.0 for image in images]
        fused = self._merge.process(float_images)
        fused = np.clip(fused * 255.0, 0, 255).astype(np.uint8)
        return fused


class PythonModuleBackend(FusionBackend):
    name = "module"

    def __init__(
        self,
        adapter_path: Path,
        model_path: Path | None = None,
        device: str | None = None,
    ) -> None:
        self.adapter_path = adapter_path
        self.model_path = model_path
        self.device = device
        self._adapter = self._load_adapter()

    def _load_adapter(self) -> Any:
        spec = importlib.util.spec_from_file_location(
            f"mef_adapter_{self.adapter_path.stem}",
            self.adapter_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load adapter from {self.adapter_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return build_adapter_object(module, self.model_path, self.device)

    def fuse(self, images: Sequence[np.ndarray], bracket_set: BracketSet) -> np.ndarray:
        target = self._adapter

        call_attempts = [
            lambda: target.fuse(
                images=images,
                bracket_set=bracket_set,
                model_path=self.model_path,
                device=self.device,
            ),
            lambda: target.fuse(images, bracket_set, self.model_path, self.device),
            lambda: target.fuse(images, bracket_set),
            lambda: target.fuse(images),
        ]

        last_error: Exception | None = None
        for attempt in call_attempts:
            try:
                result = attempt()
                return coerce_fused_image(result)
            except TypeError as exc:
                last_error = exc

        raise TypeError(
            "Adapter fuse() signature is not compatible with the pipeline"
        ) from last_error


def build_adapter_object(
    module: ModuleType,
    model_path: Path | None,
    device: str | None,
) -> Any:
    if hasattr(module, "build_backend"):
        return module.build_backend(model_path=model_path, device=device)
    if hasattr(module, "create_backend"):
        return module.create_backend(model_path=model_path, device=device)
    if hasattr(module, "Backend"):
        return module.Backend(model_path=model_path, device=device)
    if hasattr(module, "fuse"):
        return module
    raise AttributeError(
        "Adapter must define build_backend(), create_backend(), Backend, or fuse()"
    )


def coerce_fused_image(result: Any) -> np.ndarray:
    if isinstance(result, np.ndarray):
        fused = result
    elif isinstance(result, (tuple, list)) and result:
        fused = result[0]
    else:
        raise TypeError("Adapter returned an unsupported result type")

    if fused.dtype != np.uint8:
        fused = np.clip(fused, 0.0, 1.0)
        fused = (fused * 255.0).astype(np.uint8)

    if fused.ndim != 3 or fused.shape[2] != 3:
        raise ValueError(f"Expected fused image shape HxWx3, got {fused.shape}")

    return fused


def discover_metadata_files(dataset_dir: Path) -> list[Path]:
    return sorted(path for path in dataset_dir.rglob("*_meta.json") if path.is_file())


def load_bracket_set_from_metadata(metadata_path: Path) -> BracketSet:
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    directory = metadata_path.parent
    images: list[BracketImage] = []
    for image_meta in payload.get("images", []):
        image_path = directory / image_meta["filename"]
        if not image_path.exists():
            raise FileNotFoundError(
                f"Metadata references missing image {image_path}"
            )
        images.append(
            BracketImage(
                label=image_meta["label"],
                path=image_path,
                requested_exposure_us=image_meta.get("requested_exposure_us"),
                actual_exposure_us=image_meta.get("actual_exposure_us"),
                actual_gain=image_meta.get("actual_gain"),
            )
        )

    if not images:
        raise ValueError(f"Metadata file {metadata_path} does not contain any images")

    return BracketSet(
        set_id=payload.get("set_id", metadata_path.stem.replace("_meta", "")),
        timestamp=payload.get("timestamp", ""),
        directory=directory,
        images=tuple(images),
        metadata_path=metadata_path,
        bracket_mode=payload.get("bracket_mode"),
        auto_exposure_us=payload.get("auto_exposure_us"),
        auto_gain=payload.get("auto_gain"),
    )


def discover_bracket_sets(dataset_dir: Path) -> list[BracketSet]:
    metadata_files = discover_metadata_files(dataset_dir)
    if metadata_files:
        return [load_bracket_set_from_metadata(path) for path in metadata_files]
    return discover_bracket_sets_from_filenames(dataset_dir)


def discover_bracket_sets_from_filenames(dataset_dir: Path) -> list[BracketSet]:
    grouped: dict[tuple[Path, str, str], list[BracketImage]] = {}

    for image_path in sorted(dataset_dir.rglob("*")):
        if not image_path.is_file():
            continue
        match = BRACKET_IMAGE_PATTERN.fullmatch(image_path.name)
        if match is None:
            continue

        key = (
            image_path.parent,
            match.group("set_id"),
            match.group("timestamp"),
        )
        grouped.setdefault(key, []).append(
            BracketImage(
                label=match.group("label"),
                path=image_path,
                requested_exposure_us=int(match.group("exposure")),
            )
        )

    bracket_sets: list[BracketSet] = []
    for (directory, set_id, timestamp), images in sorted(grouped.items()):
        ordered = tuple(
            sorted(
                images,
                key=lambda image: (
                    image.requested_exposure_us or 0,
                    image.label,
                ),
            )
        )
        bracket_sets.append(
            BracketSet(
                set_id=set_id,
                timestamp=timestamp,
                directory=directory,
                images=ordered,
            )
        )

    return bracket_sets


def prepare_model_inputs(
    images: Sequence[np.ndarray],
    resize: tuple[int, int] | None = None,
    rgb: bool = True,
) -> np.ndarray:
    prepared: list[np.ndarray] = []

    for image in images:
        frame = image
        if resize is not None:
            width, height = resize
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prepared.append(frame.astype(np.float32) / 255.0)

    return np.stack(prepared, axis=0)


def default_output_path(output_dir: Path, bracket_set: BracketSet) -> Path:
    suffix = f"{bracket_set.set_id}_{bracket_set.timestamp}_fused.jpg"
    return output_dir / suffix


def save_fused_image(output_path: Path, fused_bgr: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), fused_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise OSError(f"Failed to write fused image to {output_path}")


def process_bracket_set(
    bracket_set: BracketSet,
    backend: FusionBackend,
    output_dir: Path,
    overwrite: bool = False,
) -> Path | None:
    output_path = default_output_path(output_dir, bracket_set)
    if output_path.exists() and not overwrite:
        return None

    fused_bgr = backend.fuse(bracket_set.load_bgr_images(), bracket_set)
    save_fused_image(output_path, fused_bgr)
    return output_path


def process_dataset(
    dataset_dir: Path,
    output_dir: Path,
    backend: FusionBackend,
    overwrite: bool = False,
    limit: int | None = None,
) -> ProcessSummary:
    bracket_sets = discover_bracket_sets(dataset_dir)
    if limit is not None:
        bracket_sets = bracket_sets[:limit]

    processed_paths: list[Path] = []
    skipped = 0

    for bracket_set in bracket_sets:
        output_path = process_bracket_set(
            bracket_set=bracket_set,
            backend=backend,
            output_dir=output_dir,
            overwrite=overwrite,
        )
        if output_path is None:
            skipped += 1
            continue
        processed_paths.append(output_path)

    return ProcessSummary(
        processed=len(processed_paths),
        skipped=skipped,
        output_paths=tuple(processed_paths),
    )


def format_bracket_set(bracket_set: BracketSet) -> str:
    labels = ", ".join(bracket_set.labels())
    return f"{bracket_set.set_id} [{labels}]"


def iter_bracket_sets(dataset_dir: Path) -> Iterable[BracketSet]:
    yield from discover_bracket_sets(dataset_dir)
