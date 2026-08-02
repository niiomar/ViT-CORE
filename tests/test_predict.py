import torch
from PIL import Image

from model_utils import build_eval_transform, build_model
from predict import collect_image_paths, predict_one


def _write_image(path, color=(10, 20, 30)):
    Image.new("RGB", (32, 32), color=color).save(path)


def test_collect_image_paths_single_file(tmp_path):
    f = tmp_path / "a.png"
    _write_image(f)
    assert collect_image_paths(str(f)) == [str(f)]


def test_collect_image_paths_directory_mixed_case(tmp_path):
    _write_image(tmp_path / "a.png")
    _write_image(tmp_path / "b.PNG")
    _write_image(tmp_path / "c.JPG")
    (tmp_path / "not_an_image.txt").write_text("x")

    paths = collect_image_paths(str(tmp_path))
    assert len(paths) == 3


def test_predict_one_returns_probability_in_unit_interval(tmp_path):
    img_path = tmp_path / "img.png"
    _write_image(img_path)

    model = build_model(num_classes=2, pretrained=False)
    model.eval()
    transform = build_eval_transform()
    device = torch.device("cpu")

    p_fake = predict_one(model, transform, str(img_path), device)
    assert 0.0 <= p_fake <= 1.0
