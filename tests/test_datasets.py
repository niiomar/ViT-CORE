import csv

import pytest
from PIL import Image

from datasets import MAX_OPEN_RETRIES, _load_samples_from_csv, _safe_open


def _write_image(path, color=(10, 20, 30)):
    Image.new("RGB", (8, 8), color=color).save(path)


def _write_csv(csv_path, rows):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        for row in rows:
            w.writerow(row)


def test_single_file_per_row(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    _write_image(root / "a.png")
    csv_path = tmp_path / "manifest.csv"
    _write_csv(csv_path, [("a.png", 0)])

    samples = _load_samples_from_csv(str(csv_path), str(root))
    assert len(samples) == 1
    assert samples[0][1] == 0


def test_directory_glob_mixed_case_extensions(tmp_path):
    # Regression test for the case-sensitive glob bug: uppercase extensions
    # (.PNG, .JPG) must not be silently dropped on case-sensitive filesystems.
    root = tmp_path / "data"
    folder = root / "clip_001"
    folder.mkdir(parents=True)
    _write_image(folder / "frame_0.png")
    _write_image(folder / "frame_1.PNG")
    _write_image(folder / "frame_2.JPG")

    csv_path = tmp_path / "manifest.csv"
    _write_csv(csv_path, [("clip_001", 1)])

    samples = _load_samples_from_csv(str(csv_path), str(root))
    assert len(samples) == 3
    assert all(label == 1 for _, label in samples)


def test_directory_with_no_images_is_skipped(tmp_path):
    root = tmp_path / "data"
    folder = root / "empty_clip"
    folder.mkdir(parents=True)
    csv_path = tmp_path / "manifest.csv"
    _write_csv(csv_path, [("empty_clip", 0)])

    samples = _load_samples_from_csv(str(csv_path), str(root))
    assert samples == []


def test_safe_open_falls_back_on_corrupt_image(tmp_path):
    good_path = tmp_path / "good.png"
    bad_path = tmp_path / "bad.png"
    _write_image(good_path)
    bad_path.write_bytes(b"not a real image")

    samples = [(str(bad_path), 0), (str(good_path), 0)]
    img = _safe_open(str(bad_path), 0, samples)
    assert img.size == (8, 8)


def test_safe_open_raises_after_exhausting_retries(tmp_path):
    bad_path = tmp_path / "bad.png"
    bad_path.write_bytes(b"not a real image")
    samples = [(str(bad_path), 0)] * (MAX_OPEN_RETRIES + 1)

    with pytest.raises(OSError):
        _safe_open(str(bad_path), 0, samples)
