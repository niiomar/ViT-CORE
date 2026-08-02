import pytest
import torch
from PIL import Image

from augmentations import AddGaussianNoise, RandomErasing, RandomJPEGCompression, get_transform


def make_test_image(size=(64, 64)):
    return Image.new("RGB", size, color=(120, 80, 200))


def test_raaug_returns_correct_tensor_shape():
    transform = get_transform("raaug")
    out = transform(make_test_image())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)


def test_dfdcselim_returns_correct_tensor_shape():
    transform = get_transform("dfdcselim")
    out = transform(make_test_image())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)


def test_unknown_transform_raises():
    with pytest.raises(ValueError):
        get_transform("not-a-real-transform")


def test_random_erasing_preserves_image_size():
    img = make_test_image((100, 100))
    out = RandomErasing(p=1.0)(img)
    assert out.size == img.size


def test_add_gaussian_noise_preserves_image_size_and_mode():
    img = make_test_image()
    out = AddGaussianNoise(std=0.1)(img)
    assert out.size == img.size
    assert out.mode == "RGB"


def test_random_jpeg_compression_returns_image_of_same_size():
    img = make_test_image()
    out = RandomJPEGCompression(quality=(30, 90))(img)
    assert out.size == img.size
