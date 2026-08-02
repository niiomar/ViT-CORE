import pytest
import torch
import torch.nn as nn

from model_utils import ModelEma, build_eval_transform, build_model, build_param_groups, validate_paths


def test_build_model_head_matches_num_classes():
    model = build_model(num_classes=2, pretrained=False)
    assert model.head.out_features == 2


def test_build_eval_transform_produces_correct_shape():
    from PIL import Image
    img = Image.new("RGB", (64, 64), color=(10, 20, 30))
    out = build_eval_transform()(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)


def test_build_param_groups_separates_bias_and_norm_from_weights():
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    groups = build_param_groups(model, weight_decay=0.05)
    decay_group = next(g for g in groups if g["weight_decay"] == 0.05)
    no_decay_group = next(g for g in groups if g["weight_decay"] == 0.0)

    # Linear.weight is 2-D -> decay; Linear.bias and LayerNorm params are 1-D -> no decay.
    assert any(p.shape == (4, 4) for p in decay_group["params"])
    assert all(p.ndim <= 1 for p in no_decay_group["params"])
    assert len(no_decay_group["params"]) >= 3  # linear.bias, ln.weight, ln.bias


def test_validate_paths_passes_when_all_exist(tmp_path):
    f = tmp_path / "exists.txt"
    f.write_text("x")
    validate_paths({"thing": str(f)})  # should not raise


def test_validate_paths_raises_listing_missing(tmp_path):
    missing = tmp_path / "nope.txt"
    with pytest.raises(FileNotFoundError, match="missing-thing"):
        validate_paths({"missing-thing": str(missing)})


def test_model_ema_moves_toward_updated_weights():
    model = nn.Linear(4, 4, bias=False)
    nn.init.zeros_(model.weight)
    ema = ModelEma(model, decay=0.9)

    nn.init.ones_(model.weight)
    ema.update(model)

    # decay=0.9 -> ema should have moved 10% of the way from 0 toward 1.
    expected = torch.full((4, 4), 0.1)
    assert torch.allclose(ema.module.weight, expected, atol=1e-6)


def test_model_ema_state_dict_round_trip():
    model = nn.Linear(4, 4)
    ema = ModelEma(model, decay=0.999)
    state = ema.state_dict()

    other = nn.Linear(4, 4)
    other_ema = ModelEma(other, decay=0.999)
    other_ema.load_state_dict(state)

    for k, v in ema.state_dict().items():
        assert torch.equal(v, other_ema.state_dict()[k])
