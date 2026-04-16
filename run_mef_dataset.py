from __future__ import annotations

import argparse
from pathlib import Path

from mef_pipeline import (
    OpenCvMertensBackend,
    PythonModuleBackend,
    discover_bracket_sets,
    format_bracket_set,
    process_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process Raspberry Pi bracket captures through the MEF pipeline"
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Folder containing Pi bracket images and optional *_meta.json files",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/mef_fused",
        help="Folder where fused outputs will be written",
    )
    parser.add_argument(
        "--backend",
        choices=("opencv", "module"),
        default="opencv",
        help="Fusion backend: OpenCV Mertens fallback or a Python MEF-Net adapter",
    )
    parser.add_argument(
        "--adapter",
        help="Path to a Python adapter file when --backend module is used",
    )
    parser.add_argument(
        "--model-path",
        help="Optional checkpoint/weights path forwarded to the adapter",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device string forwarded to the adapter, for example cpu or cuda:0",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate fused outputs even if they already exist",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N bracket sets",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered bracket sets without running fusion",
    )
    return parser.parse_args()


def build_backend(args: argparse.Namespace):
    if args.backend == "opencv":
        return OpenCvMertensBackend()

    if not args.adapter:
        raise SystemExit("--adapter is required when --backend module is selected")

    model_path = Path(args.model_path).expanduser() if args.model_path else None
    return PythonModuleBackend(
        adapter_path=Path(args.adapter).expanduser(),
        model_path=model_path,
        device=args.device,
    )


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")

    bracket_sets = discover_bracket_sets(dataset_dir)
    if not bracket_sets:
        raise SystemExit(f"No bracket sets found in {dataset_dir}")

    print(f"[*] Found {len(bracket_sets)} bracket set(s) in {dataset_dir}")
    for bracket_set in bracket_sets[:10]:
        print(f"    - {format_bracket_set(bracket_set)}")
    if len(bracket_sets) > 10:
        print(f"    ... and {len(bracket_sets) - 10} more")

    if args.list:
        return

    backend = build_backend(args)
    print(f"[*] Using backend: {backend.name}")
    summary = process_dataset(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        backend=backend,
        overwrite=args.overwrite,
        limit=args.limit,
    )

    print(
        f"[*] Completed. Processed {summary.processed} set(s), skipped {summary.skipped}."
    )
    for output_path in summary.output_paths[:10]:
        print(f"    -> {output_path}")
    if len(summary.output_paths) > 10:
        print(f"    ... and {len(summary.output_paths) - 10} more")


if __name__ == "__main__":
    main()
