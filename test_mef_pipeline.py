from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from mef_pipeline import (
    OpenCvMertensBackend,
    discover_bracket_sets,
    load_bracket_set_from_metadata,
    prepare_model_inputs,
    process_dataset,
)


def write_jpg(path: Path, value: int) -> None:
    frame = np.full((24, 32, 3), value, dtype=np.uint8)
    ok = cv2.imwrite(str(path), frame)
    if not ok:
        raise OSError(f"Failed to write test image {path}")


class MefPipelineTests(unittest.TestCase):
    def test_load_bracket_set_from_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            under = root / "set0001_20260416_101010_under_exp5000us.jpg"
            normal = root / "set0001_20260416_101010_normal_exp33000us.jpg"
            over = root / "set0001_20260416_101010_over_exp200000us.jpg"
            write_jpg(under, 20)
            write_jpg(normal, 120)
            write_jpg(over, 220)

            metadata_path = root / "set0001_20260416_101010_meta.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "set_id": "set0001",
                        "timestamp": "20260416_101010",
                        "bracket_mode": "fixed",
                        "images": [
                            {
                                "label": "under",
                                "filename": under.name,
                                "requested_exposure_us": 5000,
                                "actual_exposure_us": 5200,
                                "actual_gain": 1.1,
                            },
                            {
                                "label": "normal",
                                "filename": normal.name,
                                "requested_exposure_us": 33000,
                                "actual_exposure_us": 33100,
                                "actual_gain": 1.1,
                            },
                            {
                                "label": "over",
                                "filename": over.name,
                                "requested_exposure_us": 200000,
                                "actual_exposure_us": 198000,
                                "actual_gain": 1.1,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            bracket_set = load_bracket_set_from_metadata(metadata_path)

            self.assertEqual(bracket_set.set_id, "set0001")
            self.assertEqual(bracket_set.labels(), ("under", "normal", "over"))
            self.assertEqual(bracket_set.images[1].requested_exposure_us, 33000)
            self.assertEqual(bracket_set.metadata_path, metadata_path)

    def test_discover_bracket_sets_without_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_jpg(root / "set0007_20260416_151500_under_exp5000us.jpg", 30)
            write_jpg(root / "set0007_20260416_151500_normal_exp33000us.jpg", 90)
            write_jpg(root / "set0007_20260416_151500_over_exp200000us.jpg", 180)

            bracket_sets = discover_bracket_sets(root)

            self.assertEqual(len(bracket_sets), 1)
            self.assertEqual(bracket_sets[0].set_id, "set0007")
            self.assertEqual(bracket_sets[0].labels(), ("under", "normal", "over"))

    def test_prepare_model_inputs_returns_normalized_rgb_stack(self) -> None:
        frame = np.zeros((8, 10, 3), dtype=np.uint8)
        frame[..., 0] = 255

        stack = prepare_model_inputs([frame], resize=(4, 6), rgb=True)

        self.assertEqual(stack.shape, (1, 6, 4, 3))
        self.assertEqual(stack.dtype, np.float32)
        self.assertTrue(np.all(stack >= 0.0))
        self.assertTrue(np.all(stack <= 1.0))

    def test_process_dataset_writes_fused_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "dataset"
            output_dir = root / "out"
            dataset_dir.mkdir()

            write_jpg(dataset_dir / "set0002_20260416_121212_under_exp5000us.jpg", 20)
            write_jpg(dataset_dir / "set0002_20260416_121212_normal_exp33000us.jpg", 120)
            write_jpg(dataset_dir / "set0002_20260416_121212_over_exp200000us.jpg", 220)

            summary = process_dataset(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                backend=OpenCvMertensBackend(),
            )

            self.assertEqual(summary.processed, 1)
            self.assertEqual(summary.skipped, 0)
            self.assertEqual(len(summary.output_paths), 1)
            self.assertTrue(summary.output_paths[0].exists())


if __name__ == "__main__":
    unittest.main()
